#include <iostream>
#include <cstring> 

// Заголовочные сетевые файлы для работы с DNS и IP-адресами
#include <sys/socket.h> 
#include <netdb.h> 
#include <unistd.h>

#include <arpa/inet.h> 
#include <netinet/in.h> 


int main () {
    // Создаем сокет
    int serversoc = socket(AF_INET, SOCK_STREAM, 0);
    if (serversoc < 0) { 
        std::cout << "Ошибка при создании сокета.\n"; 
        return 1;
    }

    unsigned port = 8080;
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = INADDR_ANY; // любые адреса

    // Привязка 
    if (bind(serversoc, (sockaddr *)&address, sizeof(address)) < 0) {
        std::cout << "Ошибка привязки.\n";
        return 1;
    }

    // Прослушивание
    if (listen(serversoc, 1) < 0) {
        std::cout << "Ошибка при прослушивании: " 
        << strerror(errno) << std::endl;
        return 1;
    }

    std::cout << "Сервер запущен. Порт: ", port;

    // Принятие подключения
    sockaddr_in6 client_addr;
    socklen_t client_len = sizeof(client_addr);

    int client_socket = accept(serversoc, 
        (sockaddr *)&client_addr, 
        &client_len);
    if (client_socket < 0){
        std::cout << "Ошибка подключения.\n";
        return 1;
    }

    std::cout << "Клиент подключился.\n";

    char buffer[512];
    for(;;) {
        memset(buffer, 0, sizeof(buffer));
        ssize_t buffer_read = read(client_socket, buffer, sizeof(buffer));

        if(buffer_read < 0) {
            break;
        }

        write(client_socket, buffer, buffer_read);
    }

    // Закрываем все входы выходы
    close(client_socket);
    close(serversoc);

    return 0;
}