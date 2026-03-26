#include "kafka_producer.hpp"
#include "msk_token.hpp"
#include "../config.hpp"
#include "../utils/logger.hpp"
#include <librdkafka/rdkafkacpp.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <thread>
#include <chrono>

namespace app::kafka {

namespace {

class OauthRefreshCb : public RdKafka::OAuthBearerTokenRefreshCb {
public:
    void oauthbearer_token_refresh_cb(RdKafka::Handle* handle,
                                      const std::string& /*oauthbearer_config*/) override
    {
        if (!handle) return;
        auto& cfg = app::config::AppConfig::getInstance();
        std::string tok = generate_msk_iam_token(cfg.kafka_aws_region);
        if (tok.empty()) {
            app::utils::Logger::error("[Kafka] MSK IAM token generation failed");
            return;
        }
        std::string errstr;
        const RdKafka::ErrorCode code = handle->oauthbearer_set_token(
            tok,
            900000,
            "kafka-cluster",
            {},
            errstr);
        if (code != RdKafka::ERR_NO_ERROR) {
            app::utils::Logger::error("[Kafka] oauthbearer_set_token: " + errstr);
        }
    }
};

std::string join_brokers(const std::vector<std::string>& servers) {
    std::string s;
    for (size_t i = 0; i < servers.size(); ++i) {
        if (i) s += ',';
        s += servers[i];
    }
    return s;
}

}  // namespace

struct KafkaProducer::Impl {
    std::unique_ptr<RdKafka::Producer> producer;
    OauthRefreshCb oauth_cb;
};

KafkaProducer::KafkaProducer() : impl_(std::make_unique<Impl>()) {}
KafkaProducer::~KafkaProducer() { stop(); }

void KafkaProducer::start_with_retry() {
    auto& cfg = app::config::AppConfig::getInstance();
    if (cfg.kafka_bootstrap_servers.empty()) {
        app::utils::Logger::error("[Kafka] KAFKA_BOOTSTRAP_SERVERS not set");
        std::exit(1);
    }

    std::string errstr;
    std::unique_ptr<RdKafka::Conf> conf(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));

    if (conf->set("bootstrap.servers", join_brokers(cfg.kafka_bootstrap_servers), errstr) != RdKafka::Conf::CONF_OK) {
        app::utils::Logger::error("[Kafka] " + errstr);
        std::exit(1);
    }
    if (!cfg.kafka_client_id.empty())
        conf->set("client.id", cfg.kafka_client_id, errstr);

    conf->set("security.protocol", "SASL_SSL", errstr);
    conf->set("sasl.mechanisms", "OAUTHBEARER", errstr);
    conf->set("enable.ssl.certificate.verification", "false", errstr);
    conf->set("max.request.size", "5000000", errstr);

    if (conf->set("oauthbearer_token_refresh_cb", &impl_->oauth_cb, errstr) != RdKafka::Conf::CONF_OK) {
        app::utils::Logger::error("[Kafka] oauth cb: " + errstr);
        std::exit(1);
    }

    for (int attempt = 1; attempt <= cfg.max_retries; ++attempt) {
        std::string e2;
        impl_->producer.reset(RdKafka::Producer::create(conf.get(), e2));
        if (impl_->producer) {
            app::utils::Logger::info("[Kafka] Producer connected");
            return;
        }
        app::utils::Logger::error("[Kafka] create Producer failed (try " + std::to_string(attempt) + "): " + e2);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    app::utils::Logger::error("[Kafka] Failed to start producer after retries");
    std::exit(1);
}

void KafkaProducer::stop() {
    if (impl_ && impl_->producer) {
        impl_->producer->flush(5000);
        impl_->producer.reset();
    }
}

void KafkaProducer::produce(const std::string& topic, const app::utils::FallMessage& message) {
    if (!impl_ || !impl_->producer) {
        app::utils::Logger::error("[Kafka] produce: not started");
        std::exit(1);
    }
    nlohmann::json j = {
        {"store_id", message.store_id},
        {"moksa_camera_id", message.moksa_camera_id},
        {"s3_uri", message.s3_uri},
        {"video_uri", message.video_uri},
        {"trace_id", message.trace_id},
        {"timestamp", message.timestamp},
    };
    std::string payload = j.dump();
    const std::string& key = message.trace_id;

    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
    RdKafka::ErrorCode resp = impl_->producer->produce(
        topic,
        RdKafka::Topic::PARTITION_UA,
        RdKafka::Producer::RK_MSG_COPY,
        const_cast<char*>(payload.data()),
        payload.size(),
        key.data(),
        key.size(),
        now_ms,
        nullptr);

    if (resp != RdKafka::ERR_NO_ERROR) {
        std::string errMsg = "[Kafka] produce failed: " + std::string(RdKafka::err2str(resp));
        app::utils::Logger::error(errMsg);
        // Match Python: stop producer then re-raise to caller
        stop();
        throw std::runtime_error(errMsg);
    }
    // Match Python send_and_wait: block until broker ACKs (or timeout)
    RdKafka::ErrorCode flush_err = impl_->producer->flush(5000);
    if (flush_err != RdKafka::ERR_NO_ERROR) {
        std::string errMsg = "[Kafka] flush timed out (message may not be delivered): " +
                             std::string(RdKafka::err2str(flush_err));
        app::utils::Logger::error(errMsg);
        stop();
        throw std::runtime_error(errMsg);
    }
}

}  // namespace app::kafka
