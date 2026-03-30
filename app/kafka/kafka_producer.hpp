// Interface for the Kafka producer.
// Declares KafkaProducer, which serialises a FallEventMessage to JSON and
// publishes it to the configured Kafka/MSK topic on each confirmed fall event.

#pragma once

#include "../utils/message.hpp"
#include <memory>
#include <string>

namespace app::kafka {

class KafkaProducer {
public:
    KafkaProducer();
    ~KafkaProducer();

    /** Connects producer; retries AppConfig::max_retries on failure. Exits process if all fail (matches Python). */
    void start_with_retry();
    void stop();

    void produce(const std::string& topic, const app::utils::FallMessage& message);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace app::kafka
