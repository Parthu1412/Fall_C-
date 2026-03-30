// Integration shims — re-exports S3Client, KafkaProducer, and RabbitMQClient under
// the app::integrations namespace so msg_gen.cpp can reference them by a single
// consistent namespace without depending on each subsystem's header path directly.

#pragma once

// Pull in the real implementations we wrote earlier
#include "../../utils/aws.hpp"
#include "../../kafka/kafka_producer.hpp"
#include "../../mqtt/rabbitmq.hpp"

namespace app {
namespace integrations {
    // Create aliases so msg_gen.cpp can use them without changing its code
    using S3Client = app::utils::S3Client;
    using KafkaProducer = app::kafka::KafkaProducer;
    using RabbitMQClient = app::mqtt::RabbitMQClient;
} // namespace integrations
} // namespace app