#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// For the standard Bank Churn CSV (14 columns total)
// We skip: RowNumber(0), CustomerId(1), Surname(2)
// We take: 3 to 12 as Features (10 features)
// We take: 13 as Label (Exited)
#define INPUT_SIZE 10    
#define HIDDEN_SIZE 16   
#define OUTPUT_SIZE 2    
#define MAX_SAMPLES 5000 
#define EPOCHS 300       
#define LR 0.01          
#define BATCH_SIZE 32
#define TRAIN_SPLIT 0.8  

double X[MAX_SAMPLES][INPUT_SIZE];
int y[MAX_SAMPLES];
double W1[HIDDEN_SIZE][INPUT_SIZE], b1[HIDDEN_SIZE];
double W2[OUTPUT_SIZE][HIDDEN_SIZE], b2[OUTPUT_SIZE];
int samples = 0;

// Standard Math
double relu(double x) { return x > 0 ? x : 0; }
double relu_deriv(double x) { return x > 0 ? 1 : 0; }

void softmax(double *z, double *out) {
    double max = z[0], sum = 0; 
    for(int i=1; i<OUTPUT_SIZE; i++) if(z[i] > max) max = z[i];
    for(int i=0; i<OUTPUT_SIZE; i++) {
        out[i] = exp(z[i] - max);
        sum += out[i];
    }
    for(int i=0; i<OUTPUT_SIZE; i++) out[i] /= sum;
}

void load_data(const char *filename) {
    FILE *file = fopen(filename, "r");
    if(!file) { printf("CSV not found!\n"); exit(1); }
    char line[4096];
    fgets(line, sizeof(line), file); // Skip header

    while(fgets(line, sizeof(line), file) && samples < MAX_SAMPLES) {
        char *token = strtok(line, ",");
        int col = 0;
        int feature_idx = 0;

        while(token != NULL) {
            // SKIP columns 0, 1, and 2 (ID and Name)
            if (col >= 3 && col <= 12) {
                // If it's Geography or Gender (strings), atof returns 0. 
                // In a real ML project, you'd map "France" to 1, "Female" to 0, etc.
                X[samples][feature_idx++] = atof(token);
            } else if (col == 13) {
                y[samples] = atoi(token);
            }
            token = strtok(NULL, ",");
            col++;
        }
        samples++;
    }
    fclose(file);

    printf("DIAGNOSTIC: Loaded %d samples.\n", samples);
    printf("DIAGNOSTIC: Row 0 Features: %f, %f... Label: %d\n", X[0][0], X[0][1], y[0]);

    // Normalization
    for(int j=0; j<INPUT_SIZE; j++) {
        double min = X[0][j], max = X[0][j];
        for(int i=0; i<samples; i++) {
            if(X[i][j] < min) min = X[i][j];
            if(X[i][j] > max) max = X[i][j];
        }
        if(max == min) continue;
        for(int i=0; i<samples; i++) X[i][j] = (X[i][j] - min) / (max - min);
    }
}

// ... init_weights, train, and evaluate remain the same as the "Actual ML" version ...

void init_weights() {
    srand(42); 
    for(int i=0; i<HIDDEN_SIZE; i++) {
        for(int j=0; j<INPUT_SIZE; j++) W1[i][j] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/INPUT_SIZE);
        b1[i] = 0;
    }
    for(int i=0; i<OUTPUT_SIZE; i++) {
        for(int j=0; j<HIDDEN_SIZE; j++) W2[i][j] = ((double)rand()/RAND_MAX - 0.5) * sqrt(2.0/HIDDEN_SIZE);
        b2[i] = 0;
    }
}

void train() {
    int train_count = (int)(samples * TRAIN_SPLIT);
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_loss = 0;
        for(int s = 0; s < train_count; s += BATCH_SIZE) {
            double gW2[OUTPUT_SIZE][HIDDEN_SIZE] = {0}, gb2[OUTPUT_SIZE] = {0};
            double gW1[HIDDEN_SIZE][INPUT_SIZE] = {0}, gb1[HIDDEN_SIZE] = {0};
            int b_end = (s + BATCH_SIZE > train_count) ? train_count : s + BATCH_SIZE;

            for(int idx = s; idx < b_end; idx++) {
                double z1[HIDDEN_SIZE], a1[HIDDEN_SIZE], z2[OUTPUT_SIZE], y_hat[OUTPUT_SIZE];
                for(int i=0; i<HIDDEN_SIZE; i++) {
                    z1[i] = b1[i];
                    for(int j=0; j<INPUT_SIZE; j++) z1[i] += W1[i][j] * X[idx][j];
                    a1[i] = relu(z1[i]);
                }
                for(int i=0; i<OUTPUT_SIZE; i++) {
                    z2[i] = b2[i];
                    for(int j=0; j<HIDDEN_SIZE; j++) z2[i] += W2[i][j] * a1[j];
                }
                softmax(z2, y_hat);
                epoch_loss -= log(y_hat[y[idx]] + 1e-15);

                double dz2[OUTPUT_SIZE];
                for(int i=0; i<OUTPUT_SIZE; i++) {
                    dz2[i] = y_hat[i] - (y[idx] == i ? 1.0 : 0.0);
                    gb2[i] += dz2[i];
                    for(int j=0; j<HIDDEN_SIZE; j++) gW2[i][j] += dz2[i] * a1[j];
                }
                for(int i=0; i<HIDDEN_SIZE; i++) {
                    double err = 0;
                    for(int j=0; j<OUTPUT_SIZE; j++) err += W2[j][i] * dz2[j];
                    double dz1 = err * relu_deriv(z1[i]);
                    gb1[i] += dz1;
                    for(int j=0; j<INPUT_SIZE; j++) gW1[i][j] += dz1 * X[idx][j];
                }
            }
            for(int i=0; i<OUTPUT_SIZE; i++) {
                for(int j=0; j<HIDDEN_SIZE; j++) W2[i][j] -= (LR/BATCH_SIZE) * gW2[i][j];
                b2[i] -= (LR/BATCH_SIZE) * gb2[i];
            }
            for(int i=0; i<HIDDEN_SIZE; i++) {
                for(int j=0; j<INPUT_SIZE; j++) W1[i][j] -= (LR/BATCH_SIZE) * gW1[i][j];
                b1[i] -= (LR/BATCH_SIZE) * gb1[i];
            }
        }
        if(epoch % 50 == 0) printf("Epoch %d | Loss: %f\n", epoch, epoch_loss/train_count);
    }
}

void evaluate() {
    int train_count = (int)(samples * TRAIN_SPLIT);
    int correct = 0, total = samples - train_count;
    for(int s = train_count; s < samples; s++) {
        double z1[HIDDEN_SIZE], a1[HIDDEN_SIZE], z2[OUTPUT_SIZE], y_hat[OUTPUT_SIZE];
        for(int i=0; i<HIDDEN_SIZE; i++) {
            z1[i] = b1[i];
            for(int j=0; j<INPUT_SIZE; j++) z1[i] += W1[i][j] * X[s][j];
            a1[i] = relu(z1[i]);
        }
        for(int i=0; i<OUTPUT_SIZE; i++) {
            z2[i] = b2[i];
            for(int j=0; j<HIDDEN_SIZE; j++) z2[i] += W2[i][j] * a1[j];
        }
        softmax(z2, y_hat);
        if((y_hat[1] > y_hat[0] ? 1 : 0) == y[s]) correct++;
    }
    printf("\nAccuracy: %.2f%%\n", (double)correct/total*100);
}

int main() {
    load_data("Bank Customer Churn Prediction.csv");
    init_weights();
    train();
    evaluate();
    return 0;
}