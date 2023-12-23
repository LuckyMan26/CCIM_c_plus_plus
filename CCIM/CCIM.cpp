// CCIM.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <future>
Eigen::MatrixXd getP(int m) {
    Eigen::MatrixXd res(m,m);
    res.setZero();
   
    res(0, m-1) = 1;
    for (int i = 1; i < m ; i++) {
        res(i, i - 1) = 1;
    }
    
    return res;
}

Eigen::MatrixXd getQ(int n) {
    Eigen::MatrixXd res(n, n);
    res.setZero();

    res(n - 1, 0) = 1;
    for (int i = 1; i < n; i++) {
        res(i - 1, i) = 1;
    }
  
    return res;
}
Eigen::MatrixXd getPowerP(int exponent, Eigen::MatrixXd& P) {
    Eigen::MatrixXd res(P.rows(), P.cols());
    int rows = P.rows();
    int cols = P.cols();
    res.setZero();
    if (exponent == 0) {
        for (int i = 0; i < rows; i++) {
            res(i, i) = 1;
        }
    }
    else if (exponent > 0) {
        int e = exponent - 1;
        res(0, (rows-1-e)%rows) = 1;
        for (int i = 1; i < rows; i++) {
            if (i-1 - e < 0) {
                res(i, (i-1-e)%rows+rows) = 1;
            }
            else {
                res(i, (i - 1 - e) % rows) = 1;
            }
        }

    }

    else {
        int e = (exponent % cols + cols) - 1;
        res((rows - 1 + e)%rows, 0) = 1;
        for (int i = 1; i < rows; i++) {
            
         
                res(i, (i-1+e) % rows) = 1;
            
        }
    }
   
    return res;
}
Eigen::MatrixXd getPowerQ(int exponent, Eigen::MatrixXd& Q) {
    Eigen::MatrixXd res(Q.rows(), Q.cols());
    int rows = Q.rows();
    int cols = Q.cols();
    res.setZero();
    if (exponent == 0) {
        for (int i = 0; i < rows; i++) {
            res(i, i) = 1;
        }
    }
    else if (exponent > 0) {
        int e = exponent - 1;
        res((rows - 1 - e)%rows, 0) = 1;
        for (int i = 1; i < rows; i++) {
            if (i - e -1< 0) {
                res((i - e-1) % rows + rows, i) = 1;
            }
            else {
                res((i - 1-e)%rows, i) = 1;
            }
        }

    }

    else {
        int e = (exponent % cols + cols) - 1;
        res((rows - 1 + e)%rows, 0) = 1;
        for (int i = 1; i < rows; i++) {
            
         
                res((i +e -1)%rows, i) = 1;
            
        }
    }
  
    return res;
}

Eigen::MatrixXd correlationOperator(Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>& Z, Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>& X,int s, int t) {
    Eigen::MatrixXd res(X.rows(), X.cols());
    int rows = X.rows();
    int cols = X.cols();
    Eigen::VectorXd elem = Z(s, t);
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
           
            res(j, i) = X(j, i).dot(elem);
        }
    }
   
    return res;
}
std::unordered_map<std::pair<int,int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>> constructFundamentals(int M, int N,int m,int n, Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>& Z, Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic>& X) {
    std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>> map;
  
    Eigen::MatrixXd P = getP(M);
    Eigen::MatrixXd Q = getQ(N);

    std::vector<std::future<void>> futures;

    for (int i = 0; i < m; i++) {
        futures.push_back(std::async(std::launch::async, [i, &map, &P, &Q, &Z, &X, n]() {
            for (int j = 0; j < n; j++) {

                map[std::make_pair(i, j)] = (getPowerP(-i, P) * correlationOperator(Z, X, i, j)) * getPowerQ(-j, Q);
            }
            }));
    }

    for (auto& future : futures) {
        future.wait();
    }

    return map;
}

std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>> constructIntegralMatrix(std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>>& B,int m, int n) {
    std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>> M;
    M[std::make_pair(0, 0)] = B[std::make_pair(0,0)];
    for (int s = 1; s < m ; s++) {
        M[std::make_pair(s, 0)] = M[std::make_pair(s-1, 0)] +  B[std::make_pair(s, 0)];
    }
    for (int t = 1; t < n; t++) {
        M[std::make_pair(0, t)] = M[std::make_pair(0, t-1)] + B[std::make_pair(0, t)];
    }
    for (int s = 1; s < m; s++) {
        for (int t = 1; t < n; t++) {
            M[std::make_pair(s, t)] = M[std::make_pair(s-1, 0)] + M[std::make_pair(0, t - 1)] - M[std::make_pair(s-1, t - 1)] +B[std::make_pair(s, t)];
        }
    }
    return M;
}
Eigen::MatrixXd calculateL(int m, int n,int s,int t, std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>>& M) {
    Eigen::MatrixXd L(M[std::make_pair(0, 0)].rows(), M[std::make_pair(0, 0)].cols());
    if (s == 0 || t==0) {

        L.setZero();
        return L;
    }
    L = M[std::make_pair(m - 1, n - 1)] + M[std::make_pair(m - 1, n - t - 1)] + M[std::make_pair(m - s - 1, n - 1)] + M[std::make_pair(m -s- 1, n- t - 1)];
    return L;
}
Eigen::MatrixXd calculateG(int m, int n, int s, int t, std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>>& M) {
    Eigen::MatrixXd G(M[std::make_pair(0, 0)].rows(), M[std::make_pair(0, 0)].cols());
    if (s == 0) {

        G.setZero();
        return G;
    }
    G =  M[std::make_pair(m - 1, n - t - 1)] + M[std::make_pair(m - s - 1, n -t- 1)];
    return G;
}
Eigen::MatrixXd calculateK( int m, int n, int s, int t, std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>>& M) {
    Eigen::MatrixXd K(M[std::make_pair(0,0)].rows(), M[std::make_pair(0, 0)].cols());
    if (t == 0) {
      
        K.setZero();
        return K;
    }
    K = M[std::make_pair(m -s- 1, n  - 1)] + M[std::make_pair(m - s - 1, n - t - 1)];
    return K;
}
Eigen::MatrixXd calculateJ(int m, int n, int s, int t, std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>>& M) {
    Eigen::MatrixXd J;
   
    J =  M[std::make_pair(m - s - 1, n - t - 1)];
    return J;
}

void calculateCorrelationThread(std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>>& M,
    Eigen::MatrixXd& P, Eigen::MatrixXd& Q,
    int s, int t, int m, int n, std::vector<Eigen::VectorXd>& result, std::mutex& mutex) {
    Eigen::MatrixXd L = calculateL(m, n, s, t, M);
    Eigen::MatrixXd G = calculateG(m, n, s, t, M);
    Eigen::MatrixXd K = calculateK(m, n, s, t, M);
    Eigen::MatrixXd J = calculateJ(m, n, s, t, M);
    Eigen::MatrixXd p1 = getPowerP(m - s, P);
    Eigen::MatrixXd p2 = getPowerP(-s, P);
    Eigen::MatrixXd q1 = getPowerQ(n - t, Q);
    Eigen::MatrixXd q2 = getPowerQ(-t, Q);

    L = p1 * L * q1;
    G = p1 * G * q2;
    K = p2 * K * q1;
    J = p2 * J * q2;

    Eigen::MatrixXd res = L + G + K + J;
    Eigen::VectorXd v(Eigen::Map<Eigen::VectorXd>(res.data(), res.cols() * res.rows()));
    mutex.lock();
    result.push_back(v);
    mutex.unlock();
}
int main()
{
    cv::Mat img_X = cv::imread("E:\\test2.jpg");

    const int M = img_X.rows;
    const int N = img_X.cols;
    const int m = M / 4;
    const int n = N / 4;
    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic> X(M,N);
    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, Eigen::Dynamic> Z(m,n);
    for (int i = 0; i < m ; i++) {
        for (int j = 0; j < n; j++) {

            cv::Vec3b channels = img_X.at<cv::Vec3b>(i, j);
            Eigen::VectorXd temp(3);
            temp[0] = channels[0];
            temp[1] = channels[1];
            temp[2] = channels[2];
            Z(i, j) = temp;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            
            cv::Vec3b channels = img_X.at<cv::Vec3b>(i, j);
            Eigen::VectorXd temp(3);
            temp[0] = channels[0];
            temp[1] = channels[1];
            temp[2] = channels[2];
            X(i, j) = temp;
        }
    }
   
    Eigen::MatrixXd P = getP(M);
    Eigen::MatrixXd Q = getP(N);

    
  
    std::clock_t clock = std::clock();
    std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>>  B = constructFundamentals(M, N,m,n,Z,X);
    std::cout << (std::clock() - clock) / 1000.0 << std::endl;

    std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, boost::hash<std::pair<int, int>>> mM = constructIntegralMatrix(B, m, n);
  
    std::vector<Eigen::VectorXd> res;
    std::vector<std::thread> threads;
    
   
    std::mutex mtx;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            threads.emplace_back(calculateCorrelationThread, std::ref(mM), std::ref(P), std::ref(Q), i, j, m, n, std::ref(res),std::ref(mtx));
        }
    }

  
    for (auto& thread : threads) {
        thread.join();
    }
    std::cout << (std::clock() - clock)/1000.0 << std::endl;

}


