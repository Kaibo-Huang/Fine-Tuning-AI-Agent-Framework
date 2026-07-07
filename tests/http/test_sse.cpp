#include <catch2/catch_test_macros.hpp>

#include <string>
#include <string_view>
#include <vector>

#include "agents_framework/http/sse.hpp"

using namespace agents_framework::http;

namespace {

std::vector<SseEvent> collect(std::string_view stream, std::size_t chunk_size) {
  SseParser parser;
  std::vector<SseEvent> events;
  const auto sink = [&](const SseEvent& e) { events.push_back(e); };
  for (std::size_t i = 0; i < stream.size(); i += chunk_size) {
    parser.feed(stream.substr(i, chunk_size), sink);
  }
  return events;
}

}

TEST_CASE("SseParser parses event and data fields", "[sse]") {
  const std::string stream =
      "event: message_start\ndata: {\"a\":1}\n\n"
      "event: message_stop\ndata: {\"b\":2}\n\n";
  const auto events = collect(stream, stream.size());
  REQUIRE(events.size() == 2);
  REQUIRE(events[0].event == "message_start");
  REQUIRE(events[0].data == "{\"a\":1}");
  REQUIRE(events[1].event == "message_stop");
  REQUIRE(events[1].data == "{\"b\":2}");
}

TEST_CASE("SseParser reassembles events across arbitrary chunk boundaries", "[sse]") {
  const std::string stream =
      "event: a\ndata: hello\n\nevent: b\ndata: world\n\n";
  for (std::size_t chunk : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{7}}) {
    const auto events = collect(stream, chunk);
    REQUIRE(events.size() == 2);
    REQUIRE(events[0].data == "hello");
    REQUIRE(events[1].data == "world");
  }
}

TEST_CASE("SseParser joins multiple data lines with newlines", "[sse]") {
  const std::string stream = "data: line1\ndata: line2\n\n";
  const auto events = collect(stream, stream.size());
  REQUIRE(events.size() == 1);
  REQUIRE(events[0].event.empty());
  REQUIRE(events[0].data == "line1\nline2");
}

TEST_CASE("SseParser ignores comments and handles CRLF", "[sse]") {
  const std::string stream = ": this is a comment\r\nevent: ping\r\ndata: {}\r\n\r\n";
  const auto events = collect(stream, stream.size());
  REQUIRE(events.size() == 1);
  REQUIRE(events[0].event == "ping");
  REQUIRE(events[0].data == "{}");
}

TEST_CASE("SseParser does not dispatch data-less events", "[sse]") {
  const std::string stream = "event: only_event\n\ndata: real\n\n";
  const auto events = collect(stream, stream.size());
  REQUIRE(events.size() == 1);
  REQUIRE(events[0].data == "real");
}

TEST_CASE("SseParser strips only a single leading space after the colon", "[sse]") {
  const std::string stream = "data:  two-leading-spaces\n\n";
  const auto events = collect(stream, stream.size());
  REQUIRE(events.size() == 1);
  REQUIRE(events[0].data == " two-leading-spaces");
}
