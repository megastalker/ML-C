#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define size 50
double RELU(double x){
// return 1/(1+exp(-x));
if (x<0)return 0.01*x;
return x;
// return tanh(x);
}
double der_RELU(double x){
    // return sigmoid(x)*(1-sigmoid(x));
    if (x<0)return 0.01;
return 1;
// return 1-pow(tanh(x),2);
}
double loss(double y_true[size],double y_pred[size]){
double y = 0;
for(int i=0;i<size;i++){
    y+=pow(y_true[i]-y_pred[i],2);
}
return y/size;
}

double feed_forward(double x,double w1,double w2,double w3,double w4,double b1,double b2,double b3){
    double h1=RELU(w1*x+b1);
    double h2=RELU(w2*x+b2);
    double output=h1*w3+h2*w4+b3;
    return output;

}
int main() {
    // обучающие данные
double X[size];
double Y_true[size];
    for(int i=0;i<size;i++){
        X[i] = i-10;
        Y_true[i] = -2*(i-10) - 1;
        // Y_true[i] = pow((i-10),2);
        Y_true[i]=Y_true[i]/10.0;
    }
    double t = 0.0005;

    double w1 = ((double)rand()/RAND_MAX); 
    double b1 = ((double)rand()/RAND_MAX);

    double w2 = ((double)rand()/RAND_MAX); 
    double b2 = ((double)rand()/RAND_MAX);

    double w3 = ((double)rand()/RAND_MAX); 
    double w4 = ((double)rand()/RAND_MAX); 
    double b3 = ((double)rand()/RAND_MAX);

    

for(int j = 0;j<1000000;j++){
    double lk_w1=0;
    double lk_w2=0;
    double lk_w3=0;
    double lk_w4=0;
    double lk_b1=0;
    double lk_b2=0;
    double lk_b3=0;
    for(int i = 0;i<size;i++){
        double yt=Y_true[i];
        double x = X[i];
        double h1=RELU(w1*x+b1);
        double h2=RELU(w2*x+b2);
        double yp = feed_forward(x,w1,w2,w3,w4,b1,b2,b3);
        lk_w1+=(-2*yt+2*yp)*w3*der_RELU(x*w1+b1)*x;
        lk_w2+=(-2*yt+2*yp)*w4*der_RELU(x*w2+b2)*x;
        lk_w3+=(-2*yt+2*yp)*h1;
        lk_w4+=(-2*yt+2*yp)*h2;
        lk_b1+=(-2*yt+2*yp)*w3*der_RELU(x*w1+b1);
        lk_b2+=(-2*yt+2*yp)*w4*der_RELU(x*w2+b2);
        lk_b3+=(-2*yt+2*yp);
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
int a;
scanf("%d",&a);
printf(" %lf",feed_forward(a,w1,w2,w3,w4,b1,b2,b3)*10);
    return 0;
}