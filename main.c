#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define size 5
double sigmoid(double x){
// return 1/(1+exp(-x));
if (x<0)return 0;
return x;
}
double der_sigmoid(double x){
    // return sigmoid(x)*(1-sigmoid(x));
    if (x<0)return 0;
return 1;
}
double loss(double y_true[5],double y_pred[5]){
double y = 0;
for(int i=0;i<size;i++){
    y+=pow(y_true[i]-y_pred[i],2);
}
return y/size;
}

double feed_forward(double x,double w1,double w2,double w3,double w4,double b1,double b2,double b3){
    double h1=sigmoid(w1*x+b1);
    double h2=sigmoid(w2*x+b2);
    double output=h1*w3+h2*w4+b3;
    return output;

}
int main() {
    // обучающие данные
    double X[size] = {0, 1, 2, 3, 4};
    double Y_true[size] = {-1, -3, -5, -7, -9};
    // for(int i=0;i<size;i++){
    //     Y_true[i]=sigmoid(Y_true[i]);
    // }
    double t = 0.01;

    double w1 = ((double)rand()/RAND_MAX); 
    double b1 = ((double)rand()/RAND_MAX);

    double w2 = ((double)rand()/RAND_MAX); 
    double b2 = ((double)rand()/RAND_MAX);

    double w3 = ((double)rand()/RAND_MAX); 
    double w4 = ((double)rand()/RAND_MAX); 
    double b3 = ((double)rand()/RAND_MAX);

    

for(int j = 0;j<1000;j++){
    double lk_w1=0;
    double lk_w2=0;
    double lk_w3=0;
    double lk_w4=0;
    double lk_b1=0;
    double lk_b2=0;
    double lk_b3=0;
    for(int i = 0;i<5;i++){
        double yt=Y_true[i];
        double x = X[i];
        double h1=sigmoid(w1*x+b1);
        double h2=sigmoid(w2*x+b2);
        double yp = feed_forward(x,w1,w2,w3,w4,b1,b2,b3);
        lk_w1+=(-2*yt+2*yp)*der_sigmoid(h1*w3+h2*w4+b3)*w3*der_sigmoid(x*w1+b1)*x;
        lk_w2+=(-2*yt+2*yp)*der_sigmoid(h1*w3+h2*w4+b3)*w4*der_sigmoid(x*w2+b2)*x;
        lk_w3+=(-2*yt+2*yp)*der_sigmoid(h1*w3+h2*w4+b3)*h1;
        lk_w4+=(-2*yt+2*yp)*der_sigmoid(h1*w3+h2*w4+b3)*h2;
        lk_b1+=(-2*yt+2*yp)*der_sigmoid(h1*w3+h2*w4+b3)*w3*der_sigmoid(x*w1+b1);
        lk_b2+=(-2*yt+2*yp)*der_sigmoid(h1*w3+h2*w4+b3)*w4*der_sigmoid(x*w2+b2);
        lk_b3+=(-2*yt+2*yp)*der_sigmoid(h1*w3+h2*w4+b3);
    }
    lk_w1/=size;
    lk_w2/=size;        
    lk_w3/=size;
    lk_w4/=size;
    lk_b1/=size;
    lk_b2/=size;
    lk_b3/=size;
    w1-=t*lk_w1;
    w2-=t*lk_w2;
    w3-=t*lk_w3;
    w4-=t*lk_w4;
    b1-=t*lk_b1;
    b2-=t*lk_b2;
    b3-=t*lk_b3;
}
printf("10 %lf",feed_forward(10,w1,w2,w3,w4,b1,b2,b3));

    return 0;
}