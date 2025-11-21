#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define size 50
#define size_hidden 10
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

double feed_forward(double x,double w1[size_hidden],double w2[size_hidden],double b1[size_hidden], double b2){
    double h[size_hidden];
    for(int i=0;i<size_hidden;i++){
        h[i]=RELU(w1[i]*x+b1[i]);
    }
    double output=0;
    for(int i=0;i<size_hidden;i++){
        output+=h[i]*w2[i];
    }
    output+=b2;
    return output;
}

int main() {
    double X[size];
double Y_true[size];
    for(int i=0;i<size;i++){
        X[i] = i-10;
        // Y_true[i] = -2*(i-10) - 1;
        Y_true[i] = pow((i-10),2);
        Y_true[i]=Y_true[i];
    }
    double t = 0.0005;
    double w1[size_hidden];
    double b1[size_hidden];
    double w2[size_hidden];
    double b2 = ((double)rand()/RAND_MAX);
for(int i=0;i<size_hidden;i++){
    w1[i] = ((double)rand()/RAND_MAX); 
    b1[i] = ((double)rand()/RAND_MAX);
    w2[i] = ((double)rand()/RAND_MAX); 
}
for(int j = 0;j<1000000;j++){
    double lk_w1[size_hidden]={0};
    double lk_w2[size_hidden]={0};
    double lk_b1[size_hidden]={0};
    double lk_b2=0;
    double h[size_hidden];
    for(int n=0;n<size;n++){
        double x = X[n];
        double yt = Y_true[n];
        double yp = feed_forward(x,w1,w2,b1,b2);
        for(int i=0;i<size_hidden;i++){
            h[i]=RELU(w1[i]*x+b1[i]);
    }
    double lk_t = 2*(yp-yt);
    for(int i=0;i<size_hidden;i++){
        lk_w1[i]+=lk_t*w2[i]*der_RELU(w1[i]*x+b1[i])*x;
        lk_w2[i]+=lk_t*h[i];
        lk_b1[i]+=lk_t*der_RELU(w1[i]*x+b1[i]);
    }  
    lk_b2 += lk_t;
    }
    for(int i = 0;i<size_hidden;i++){
        w1[i]-=t*lk_w1[i]/size;
        w2[i]-=t*lk_w2[i]/size;
        b1[i]-=t*lk_b1[i]/size;
    }
    b2-=t*lk_b2/size;
}
for(int i=0;i<100;i++){
    double a = 0;
    printf("enter x: ");
    scanf("%lf",&a);
    printf(" %lf\n",feed_forward(a,w1,w2,b1,b2));
}
}