#include <ctime>
#include <iostream>
#include <string>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <unistd.h>
#include <vector>
#include <atomic>
using boost::asio::ip::udp;

std::atomic<int> atomic_command;
std::vector<double> command {0, 0, 0, 0};
class udp_server
{
public:
  udp_server(boost::asio::io_service& io_service, int port)
    : socket_(io_service, udp::endpoint(udp::v4(), port))
  {
    std::cout << "Listening on port " << port << std::endl;
    // start_receive();
  }
  std::string start_receive()
  {
     socket_.async_receive_from(
        boost::asio::buffer(recv_buffer_), remote_endpoint_,
        boost::bind(&udp_server::handle_receive, this,
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred));
    //   std::cout << "Bytes received: " << boost::asio::placeholders::bytes_transferred << std::endl;
    //   std::cout << "Received: " << std::string(reinterpret_cast<const char*>(recv_buffer_.data())) << std::endl;
    // std::string msg = std::string(reinterpret_cast<const char*>(recv_buffer_.data()));
    // return msg;
  }
private:
  void handle_receive(const boost::system::error_code& error,
      std::size_t bytes_transferred)
  {
    std::string msg = std::string(reinterpret_cast<const char*>(recv_buffer_.data()));
    std::cout << "Received: " << msg << std::endl;
    if (!error || error == boost::asio::error::message_size)
    {
      start_receive();
    }
    std::cout << "done handle_receive" << std::endl;
  }
  udp::socket socket_;
  udp::endpoint remote_endpoint_;
  boost::array<char, 1024> recv_buffer_;
};
int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cout << "Usage: udpserver [listen port]\n";
    std::cout << "Example: udpserver 8094\n";
    return 1;
  }
  unsigned int port = atoi(argv[1]);
  setvbuf(stdout, NULL, _IONBF, 0);
//   try
//   { 
    boost::asio::io_service io_service;
    udp_server server(io_service, port);
    boost::asio::io_service::work(io_service);
    io_service.run();

    server.start_receive();
    std::future<std::string> future_ret = std::async(std::launch::async, [&]()->std::string{return server.start_receive();});
// //   }
// //   catch (std::exception& e)
// //   {
// //     std::cerr << e.what() << std::endl;
// //   }

  std::future_status status; 
  
  while(1){
      status = future_ret.wait_for(std::chrono::milliseconds(1));
      if (status == std::future_status::ready) {
        std::string ret = future_ret.get();
        std::cout << "ready: " << ret << std::endl;
        future_ret = std::async(std::launch::async, [&]()->std::string{return server.start_receive();});
      }
    //   std::cout << "atomic: " << atomic_command << std::endl;
    //   std::cout << "command ";
    //   for (auto &i: command) {
    //     std::cout << i << " ";
    //   }
    //   std::cout << std::endl;
  }
}
