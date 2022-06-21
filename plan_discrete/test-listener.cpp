#include <ctime>
#include <iostream>
#include <string>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <unistd.h>
#include <future>
#include <atomic>
using boost::asio::ip::udp;

std::vector<double> command {0, 0, 0, 0};
std::atomic<int> udp_command(0);
class udp_server
{
public:
    udp_server(boost::asio::io_service &io_service, int port)
        : socket_(io_service, udp::endpoint(udp::v4(), port))
    {
        std::cout << "Listening on port " << port << std::endl;
        start_receive();
    }

private:
    void start_receive()
    {
        socket_.async_receive_from(
            boost::asio::buffer(recv_buffer_), remote_endpoint_,
            boost::bind(&udp_server::handle_receive, this,
                        boost::asio::placeholders::error,
                        boost::asio::placeholders::bytes_transferred));
    }
    void handle_receive(const boost::system::error_code &error,
                        std::size_t bytes_transferred)
    {
        std::cout << "Bytes received: " << boost::asio::placeholders::bytes_transferred << std::endl;
        std::string msg = std::string(reinterpret_cast<const char *>(recv_buffer_.data()));
        std::cout << "Received: " << msg << std::endl;
        // std::fill(command.begin(), command.end(), 0);
        udp_command = std::stoi(msg.substr(0, 1));
        recv_buffer_.assign(0);
        if (!error || error == boost::asio::error::message_size)
        {
            start_receive();
        }
    }
    udp::socket socket_;
    udp::endpoint remote_endpoint_;
    boost::array<char, 1000> recv_buffer_;
};
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: udpserver [listen port]\n";
        std::cout << "Example: udpserver 8094\n";
        return 1;
    }
    unsigned int port = atoi(argv[1]);
    setvbuf(stdout, NULL, _IONBF, 0);

    boost::asio::io_service io_service;
    udp_server server(io_service, port);
    // boost::thread t(boost::bind(&boost::asio::io_service::run, &io_service));
    // [&](){io_service.run();}();
    auto future_ret = std::async(std::launch::async, [&](){io_service.run();});
    // t.detach();

    while(1){
        sleep(1);
        std::cout << "command ";
        std::cout << udp_command; 
        // for (auto &i: command) {
        //     std::cout << i << " ";
        // }
        std::cout << std::endl;
    }
    return 0;
}