#include <stdio.h>
#include <string.h>

#define MAX_NAME 50

typedef struct {
    char name[MAX_NAME];
    int accountNumber;
    double balance;
    double loanAmount;
    double interestRate;
} BankAccount;

void createAccount(BankAccount *acc);
void deposit(BankAccount *acc);
void withdraw(BankAccount *acc);
void takeLoan(BankAccount *acc);
void payLoan(BankAccount *acc);
void applyMonthlyInterest(BankAccount *acc);
void displayAccount(BankAccount acc);

int main() {
    BankAccount account;
    int choice;
    int accountCreated = 0;

    while (1) {
        printf("\n===== BANK SYSTEM =====\n");
        printf("1. Create Account\n");
        printf("2. Deposit\n");
        printf("3. Withdraw\n");
        printf("4. Take Loan\n");
        printf("5. Pay Loan\n");
        printf("6. Apply Monthly Loan Interest\n");
        printf("7. Display Account Details\n");
        printf("8. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {

            case 1:
                createAccount(&account);
                accountCreated = 1;
                break;

            case 2:
                if (accountCreated)
                    deposit(&account);
                else
                    printf("Create an account first!\n");
                break;

            case 3:
                if (accountCreated)
                    withdraw(&account);
                else
                    printf("Create an account first!\n");
                break;

            case 4:
                if (accountCreated)
                    takeLoan(&account);
                else
                    printf("Create an account first!\n");
                break;

            case 5:
                if (accountCreated)
                    payLoan(&account);
                else
                    printf("Create an account first!\n");
                break;

            case 6:
                if (accountCreated)
                    applyMonthlyInterest(&account);
                else
                    printf("Create an account first!\n");
                break;

            case 7:
                if (accountCreated)
                    displayAccount(account);
                else
                    printf("Create an account first!\n");
                break;

            case 8:
                printf("Thank you for using the Bank System!\n");
                return 0;

            default:
                printf("Invalid choice. Try again.\n");
        }
    }

    return 0;
}


void createAccount(BankAccount *acc) {
    printf("Enter account holder name: ");
    scanf(" %[^\n]", acc->name);

    printf("Enter account number: ");
    scanf("%d", &acc->accountNumber);

    acc->balance = 0.0;
    acc->loanAmount = 0.0;
    acc->interestRate = 0.05;

    printf("Account created successfully!\n");
}

void deposit(BankAccount *acc) {
    double amount;
    printf("Enter amount to deposit: ");
    scanf("%lf", &amount);

    if (amount > 0) {
        acc->balance += amount;
        printf("Deposit successful!\n");
    } else {
        printf("Invalid deposit amount!\n");
    }
}

void withdraw(BankAccount *acc) {
    double amount;
    printf("Enter amount to withdraw: ");
    scanf("%lf", &amount);

    if (amount > 0 && amount <= acc->balance) {
        acc->balance -= amount;
        printf("Withdrawal successful!\n");
    } else {
        printf("Insufficient balance or invalid amount!\n");
    }
}

void takeLoan(BankAccount *acc) {
    double amount;

    printf("Enter loan amount: ");
    scanf("%lf", &amount);

    if (amount > 0) {
        double interest = amount * acc->interestRate;
        acc->loanAmount += amount + interest;
        acc->balance += amount;

        printf("Loan approved!\n");
        printf("Interest added: %.2lf\n", interest);
        printf("Total loan to repay: %.2lf\n", acc->loanAmount);
    } else {
        printf("Invalid loan amount!\n");
    }
}

void payLoan(BankAccount *acc) {
    double amount;
    double minimumPayment;

    if (acc->loanAmount <= 0) {
        printf("No outstanding loan!\n");
        return;
    }

    minimumPayment = acc->loanAmount * 0.10;  // 10% minimum payment

    printf("Outstanding loan: %.2lf\n", acc->loanAmount);
    printf("Minimum payment required (10%%): %.2lf\n", minimumPayment);
    printf("Enter payment amount: ");
    scanf("%lf", &amount);

    if (amount < minimumPayment) {
        printf("Payment too low! You must pay at least %.2lf\n", minimumPayment);
        return;
    }

    if (amount > acc->balance) {
        printf("Insufficient balance!\n");
        return;
    }

    if (amount > acc->loanAmount)
        amount = acc->loanAmount;

    acc->loanAmount -= amount;
    acc->balance -= amount;

    printf("Loan payment successful!\n");
    printf("Remaining loan: %.2lf\n", acc->loanAmount);
}

void applyMonthlyInterest(BankAccount *acc) {
    if (acc->loanAmount > 0) {
        double interest = acc->loanAmount * acc->interestRate;
        acc->loanAmount += interest;

        printf("Monthly interest applied: %.2lf\n", interest);
        printf("New total loan: %.2lf\n", acc->loanAmount);
    } else {
        printf("No loan to apply interest on.\n");
    }
}

void displayAccount(BankAccount acc) {
    printf("\n===== ACCOUNT DETAILS =====\n");
    printf("Name: %s\n", acc.name);
    printf("Account Number: %d\n", acc.accountNumber);
    printf("Balance: %.2lf\n", acc.balance);
    printf("Outstanding Loan: %.2lf\n", acc.loanAmount);
}