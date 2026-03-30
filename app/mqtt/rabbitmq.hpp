// RabbitMQ client — publishes fall-event notification messages to a RabbitMQ
// exchange via AMQP. Uses SimpleAmqpClient and reconnects automatically on
// connection loss, matching the Python mqtt/rabitmq.py behaviour.

#pragma once

#include <SimpleAmqpClient/SimpleAmqpClient.h>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <chrono>
#include <cstdlib>
#include "../utils/message.hpp"
#include "../utils/logger.hpp"
#include "../config.hpp"

using json = nlohmann::json;

namespace app {
namespace mqtt {

class RabbitMQClient {
public:
    RabbitMQClient() {
        auto& config = app::config::AppConfig::getInstance();
        max_retries_ = config.max_retries;
    }

    // Creates a single attempt channel
    // Throws on failure (caller handles retry).
    void connect() {
        auto& config = app::config::AppConfig::getInstance();
        if (config.rabbitmq_host.empty()) {
            app::utils::Logger::info("[RabbitMQ] RABBITMQ_HOST not set – RabbitMQ publishing disabled.");
            return;
        }
        if (config.rabbitmq_use_ssl) {
            try {
                // TLS for AWS MQ: CERT_NONE
                const char* ca_path = "/etc/ssl/certs/ca-certificates.crt";
                channel_ = AmqpClient::Channel::CreateSecure(
                    ca_path,
                    config.rabbitmq_host,
                    "", "",
                    config.rabbitmq_port,
                    config.rabbitmq_user,
                    config.rabbitmq_pass,
                    "/", 131072,
                    false  // verify_hostname_and_peer = false 
                );
            } catch (const std::exception& e) {
                std::string err(e.what());
                if (err.find("SSL support") != std::string::npos || err.find("SSL") != std::string::npos) {
                    app::utils::Logger::error("[RabbitMQ] RABBITMQ_USE_SSL=1 but SimpleAmqpClient was built "
                        "without SSL. Trying non-SSL fallback.");
                    channel_ = AmqpClient::Channel::Create(
                        config.rabbitmq_host, config.rabbitmq_port,
                        config.rabbitmq_user, config.rabbitmq_pass);
                } else {
                    throw;
                }
            }
        } else {
            channel_ = AmqpClient::Channel::Create(
                config.rabbitmq_host,
                config.rabbitmq_port,
                config.rabbitmq_user,
                config.rabbitmq_pass
            );
        }
        app::utils::Logger::info("[RabbitMQ] Connected to " + config.rabbitmq_host + ":" +
            std::to_string(config.rabbitmq_port) + (config.rabbitmq_use_ssl ? " (SSL)" : ""));
    }

    // Retries up to max_retries, then exits
    void connect_with_retry() {
        auto& config = app::config::AppConfig::getInstance();
        int retry_count = 0;
        while (retry_count < max_retries_) {
            try {
                connect();
                app::utils::Logger::info("[RabbitMQ] Connection successful");
                return;
            } catch (const std::exception& e) {
                retry_count++;
                app::utils::Logger::error("[RabbitMQ] Connection failed (retry " +
                    std::to_string(retry_count) + "/" + std::to_string(max_retries_) +
                    "): " + e.what());
                if (retry_count < max_retries_)
                    std::this_thread::sleep_for(std::chrono::seconds(5));  // Match Python retry_delay=5
            }
        }
        // Sys.exit(1) after max retries
        app::utils::Logger::error("[RabbitMQ] Max retries reached. Exiting application.");
        std::exit(1);
    }

    bool is_connected() const { return channel_ != nullptr; }

    // Reconnects with retry on stale connection, retries publish once
    void publish(const std::string& queue_name, const app::utils::FallMessage& message) {
        // Check if connection needs to be established
        if (!channel_) {
            app::utils::Logger::warning("[RabbitMQ] Cannot publish, channel is null. Reconnecting...");
            connect_with_retry();
            if (!channel_) return;
        }

        json j = message;
        std::string payload = j.dump();

        try {
            channel_->DeclareQueue(queue_name, false, true, false, false);
            auto amqp_msg = AmqpClient::BasicMessage::Create(payload);
            amqp_msg->DeliveryMode(AmqpClient::BasicMessage::dm_persistent);
            channel_->BasicPublish("", queue_name, amqp_msg);
            app::utils::Logger::info("[RabbitMQ] Published trace_id: " + message.trace_id +
                " to queue: " + queue_name);

        } catch (const std::exception& e) {
            // Stale connection — reconnect and retry once
            app::utils::Logger::warning("[RabbitMQ] Publish failed (stale connection), reconnecting and retrying | error=" +
                std::string(e.what()) + " | queue=" + queue_name);
            channel_ = nullptr;
            connect_with_retry();

            // Retry publish once after reconnect
            channel_->DeclareQueue(queue_name, false, true, false, false);
            auto amqp_msg2 = AmqpClient::BasicMessage::Create(payload);
            amqp_msg2->DeliveryMode(AmqpClient::BasicMessage::dm_persistent);
            channel_->BasicPublish("", queue_name, amqp_msg2);
            app::utils::Logger::info("[RabbitMQ] Published after reconnect trace_id: " + message.trace_id +
                " to queue: " + queue_name);
        }
    }

    void close() {
        // close() — release channel
        channel_ = nullptr;
        app::utils::Logger::info("[RabbitMQ] Connection closed");
    }

private:
    AmqpClient::Channel::ptr_t channel_;
    int max_retries_ = 3;
};

} // namespace mqtt
} // namespace app