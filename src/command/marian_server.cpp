#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"
#include "common/timer.h"
#include "common/utils.h"
#include "models/bert.h"

#include "3rd_party/simple-websocket-server/server_ws.hpp"

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WSServer;

int main(int argc, char **argv) {
  using namespace marian;

  // Initialize translation task
  auto options = parseOptions(argc, argv, cli::mode::server, true);

  // Initialize web server
  WSServer server;
  server.config.port = (short)options->get<size_t>("port", 8080);

  auto translateTask = New<TranslateService<BeamSearch>>(options);
  auto &translate = server.endpoint["^/translate/?$"];

  translate.on_message = [&translateTask](Ptr<WSServer::Connection> connection,
                                          Ptr<WSServer::Message> message) {
    // Get input text
    auto inputText = message->string();
    auto sendStream = std::make_shared<WSServer::SendStream>();

    // Translate
    timer::Timer timer;
    auto outputText = translateTask->run(inputText);
    LOG(info, "Best translation: {}", outputText);
    *sendStream << outputText << std::endl;
    LOG(info, "Translation took: {:.5f}s", timer.elapsed());

    // Send translation back
    connection->send(sendStream, [](const SimpleWeb::error_code &ec) {
      if(ec) {
        LOG(error, "Error sending message: ({}) {}", ec.value(), ec.message());
      }
    });
  };

  // Error Codes for error code meanings
  // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
  translate.on_error = [](Ptr<WSServer::Connection> /*connection*/,
                          const SimpleWeb::error_code &ec) {
    LOG(error, "Connection error: ({}) {}", ec.value(), ec.message());
  };

  // Create endpoint for BERT prediction
  auto taggingTask = New<TaggingService>(options);

  if(options->hasAndNotEmpty("tagging-model")) {
    auto &tagging = server.endpoint["^/tag/?$"];

    tagging.on_message = [&taggingTask](Ptr<WSServer::Connection> connection,
                                        Ptr<WSServer::Message> message) {
      // Get input text
      auto inputText = message->string();
      auto sendStream = std::make_shared<WSServer::SendStream>();

      // Tagging
      timer::Timer timer;
      auto outputText = taggingTask->run(inputText);
      LOG(info, "Tags: {}", outputText);
      *sendStream << outputText << std::endl;
      LOG(info, "Tagging took: {:.5f}s", timer.elapsed());

      // Send the results back
      connection->send(sendStream, [](const SimpleWeb::error_code &ec) {
        if(ec) {
          LOG(error, "Error sending message: ({}) {}", ec.value(), ec.message());
        }
      });
    };

    tagging.on_error = [](Ptr<WSServer::Connection> /*connection*/,
                          const SimpleWeb::error_code &ec) {
      LOG(error, "Connection error: ({}) {}", ec.value(), ec.message());
    };
  }

  // Start server thread
  std::thread serverThread([&server]() {
    LOG(info, "Server is listening on port {}", server.config.port);
    server.start();
  });

  serverThread.join();

  return 0;
}
