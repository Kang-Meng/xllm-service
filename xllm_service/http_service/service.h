#pragma once

#include <brpc/channel.h>

#include <iostream>
#include <mutex>

#include "chat.pb.h"
#include "common/call_data.h"
#include "common/threadpool.h"
#include "common/types.h"
#include "completion.pb.h"
#include "request_tracer.h"
#include "xllm_http_service.pb.h"

namespace xllm_service {

using CompletionCallData = StreamCallData<llm::proto::CompletionRequest,
                                          llm::proto::CompletionResponse>;

using ChatCallData =
    StreamCallData<llm::proto::ChatRequest, llm::proto::ChatResponse>;

class XllmRpcServiceImpl;

class XllmHttpServiceImpl : public proto::XllmHttpService {
 public:
  XllmHttpServiceImpl(const HttpServiceConfig& config);
  XllmHttpServiceImpl(std::shared_ptr<XllmRpcServiceImpl> rpc_service,
                      const HttpServiceConfig& config);
  ~XllmHttpServiceImpl();

  void Hello(::google::protobuf::RpcController* controller,
             const proto::HttpHelloRequest* request,
             proto::HttpHelloResponse* response,
             ::google::protobuf::Closure* done) override;

  void Completions(::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done) override;

  void ChatCompletions(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void Embeddings(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void Models(::google::protobuf::RpcController* controller,
              const proto::HttpRequest* request,
              proto::HttpResponse* response,
              ::google::protobuf::Closure* done) override;

  void Metrics(::google::protobuf::RpcController* controller,
               const proto::HttpRequest* request,
               proto::HttpResponse* response,
               ::google::protobuf::Closure* done) override;

 private:
  bool create_channel(const std::string& target_uri);

  void post_serving(const std::string& serving_method,
                    ::google::protobuf::RpcController* controller,
                    const proto::HttpRequest* request,
                    proto::HttpResponse* response,
                    ::google::protobuf::Closure* done);

  template <typename T>
  void handle(std::shared_ptr<T> call_data,
              const std::string& req_attachment,
              const std::string& service_request_id,
              bool stream,
              const std::string& model,
              bool include_usage,
              const std::string& target_uri,
              const std::string& method);

  void handle_v1_chat_completions(std::shared_ptr<ChatCallData> call_data,
                                  const std::string& req_attachment,
                                  const std::string& service_request_id,
                                  bool stream,
                                  const std::string& model,
                                  bool include_usage,
                                  const std::string& target_uri);

  void handle_v1_completions(std::shared_ptr<CompletionCallData> call_data,
                             const std::string& req_attachment,
                             const std::string& service_request_id,
                             bool stream,
                             const std::string& model,
                             bool include_usage,
                             const std::string& target_uri);

  void get_serving(const std::string& serving_method,
                   ::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done);

 private:
  bool initialized_ = false;
  HttpServiceConfig config_;

  std::shared_ptr<XllmRpcServiceImpl> rpc_service_;

  std::unique_ptr<RequestTracer> request_tracer_;

  std::unique_ptr<ThreadPool> thread_pool_;

  // In disagg pd mode, we support receive generated token from
  // prefill or from decode directly.
  // 1.
  // [service] ---req---> [prefill] ---req---> [decode]
  // [service] <---first resp--- [prefill] ---first resp---> [decode]
  // [service] <---resp--- [prefill] <---resp--- [decode]
  //
  // 2.
  // [service] ---req---> [prefill] ---req---> [decode]
  // [service] <---first resp-- [prefill] --first resp---> [decode]
  // [service] <---resp-- [decode]
  //
  bool enable_decode_response_to_service_ = false;
};

}  // namespace xllm_service
