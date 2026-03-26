#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 20
#define HIDDEN_SIZE 12
#define OUTPUT_SIZE 3
#define MAX_SAMPLES 2000
#define EPOCHS 800
#define LR 0.01

double X[MAX_SAMPLES][INPUT_SIZE];
int y[MAX_SAMPLES];

double W1[HIDDEN_SIZE][INPUT_SIZE];
double b1[HIDDEN_SIZE];

double W2[OUTPUT_SIZE][HIDDEN_SIZE];
double b2[OUTPUT_SIZE];

int samples = 0;

double relu(double x){ return x > 0 ? x : 0; }
double relu_deriv(double x){ return x > 0 ? 1 : 0; }

void softmax(double *z, double *out){
    double max = z[0];
    for(int i=1;i<OUTPUT_SIZE;i++)
        if(z[i] > max) max = z[i];

    double sum = 0;
    for(int i=0;i<OUTPUT_SIZE;i++){
        out[i] = exp(z[i] - max);
        sum += out[i];
    }

    for(int i=0;i<OUTPUT_SIZE;i++)
        out[i] /= sum;
}

void init_weights(){
    srand(time(NULL));

    for(int i=0;i<HIDDEN_SIZE;i++){
        for(int j=0;j<INPUT_SIZE;j++)
            W1[i][j] = ((double)rand()/RAND_MAX)*0.01;
        b1[i] = 0;
    }

    for(int i=0;i<OUTPUT_SIZE;i++){
        for(int j=0;j<HIDDEN_SIZE;j++)
            W2[i][j] = ((double)rand()/RAND_MAX)*0.01;
        b2[i] = 0;
    }
}

void load_csv(const char *filename){

    FILE *file = fopen(filename, "r");
    if(!file){
        printf("Error opening file\n");
        exit(1);
    }

    char line[4096];

    while(fgets(line, sizeof(line), file)){
        char *token = strtok(line, ",");
        int col = 0;

        while(token){
            if(col < INPUT_SIZE)
                X[samples][col] = atof(token);
            else
                y[samples] = atoi(token);

            token = strtok(NULL, ",");
            col++;
        }

        samples++;
    }

    fclose(file);
}

void train(){

    for(int epoch=0; epoch<EPOCHS; epoch++){

        for(int s=0; s<samples; s++){

            double z1[HIDDEN_SIZE], a1[HIDDEN_SIZE];

            for(int i=0;i<HIDDEN_SIZE;i++){
                z1[i] = b1[i];
                for(int j=0;j<INPUT_SIZE;j++)
                    z1[i] += W1[i][j] * X[s][j];
                a1[i] = relu(z1[i]);
            }

            double z2[OUTPUT_SIZE], y_hat[OUTPUT_SIZE];

            for(int i=0;i<OUTPUT_SIZE;i++){
                z2[i] = b2[i];
                for(int j=0;j<HIDDEN_SIZE;j++)
                    z2[i] += W2[i][j] * a1[j];
            }

            softmax(z2, y_hat);

            double dz2[OUTPUT_SIZE];
            for(int i=0;i<OUTPUT_SIZE;i++)
                dz2[i] = y_hat[i] - (y[s] == i ? 1.0 : 0.0);

            for(int i=0;i<OUTPUT_SIZE;i++){
                for(int j=0;j<HIDDEN_SIZE;j++)
                    W2[i][j] -= LR * dz2[i] * a1[j];
                b2[i] -= LR * dz2[i];
            }

            double dz1[HIDDEN_SIZE];

            for(int i=0;i<HIDDEN_SIZE;i++){
                double sum=0;
                for(int j=0;j<OUTPUT_SIZE;j++)
                    sum += W2[j][i] * dz2[j];
                dz1[i] = sum * relu_deriv(z1[i]);
            }

            for(int i=0;i<HIDDEN_SIZE;i++){
                for(int j=0;j<INPUT_SIZE;j++)
                    W1[i][j] -= LR * dz1[i] * X[s][j];
                b1[i] -= LR * dz1[i];
            }
        }
    }
}

void evaluate(){
    int correct=0;

    for(int s=0; s<samples; s++){

        double z1[HIDDEN_SIZE], a1[HIDDEN_SIZE];

        for(int i=0;i<HIDDEN_SIZE;i++){
            z1[i] = b1[i];
            for(int j=0;j<INPUT_SIZE;j++)
                z1[i] += W1[i][j] * X[s][j];
            a1[i] = relu(z1[i]);
        }

        double z2[OUTPUT_SIZE], y_hat[OUTPUT_SIZE];

        for(int i=0;i<OUTPUT_SIZE;i++){
            z2[i] = b2[i];
            for(int j=0;j<HIDDEN_SIZE;j++)
                z2[i] += W2[i][j] * a1[j];
        }

        softmax(z2, y_hat);

        int pred=0;
        for(int i=1;i<OUTPUT_SIZE;i++)
            if(y_hat[i] > y_hat[pred])
                pred=i;

        if(pred == y[s])
            correct++;
    }

    printf("Accuracy: %.2f%%\n", (double)correct/samples*100);
}

int main(){

    load_csv("data.csv"); 
    init_weights();
    train();
    evaluate();

    return 0;
}