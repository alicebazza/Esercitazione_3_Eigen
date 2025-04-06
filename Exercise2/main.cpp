#include <iostream>
#include <Eigen/Dense>
#include <iomanip>

using namespace std;
using namespace Eigen;

// Funzione per calcolare la soluzione di Ax=b tramite la decomposizione PALU
VectorXd solve_palu(const MatrixXd& A, const VectorXd& b) {
    FullPivLU<MatrixXd> lu(A);
    return lu.solve(b);
}

// Funzione per calcolare la soluzione di Ax=b tramite la decomposizione QR
VectorXd solve_qr(const MatrixXd& A, const VectorXd& b) {
    HouseholderQR<MatrixXd> qr(A);
    return qr.solve(b);
}

// Funzione per calcolare l'errore relativo
double relative_error(const VectorXd& x, const VectorXd& x_ex) {
    return (x - x_ex).norm() / x_ex.norm();
}

int main()
{
	// Sistema 1
    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    // Sistema 2
    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    // Sistema 3
    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    // Soluzione esatta
    VectorXd x_ex(2);
    x_ex << -1.0, -1.0;	
    
    // Soluzione con PALU
    VectorXd x_palu1 = solve_palu(A1, b1);
    VectorXd x_palu2 = solve_palu(A2, b2);
    VectorXd x_palu3 = solve_palu(A3, b3);

    // Soluzione con QR
    VectorXd x_qr1 = solve_qr(A1, b1);
    VectorXd x_qr2 = solve_qr(A2, b2);
    VectorXd x_qr3 = solve_qr(A3, b3);	
    
    // Calcolo degli errori relativi
    cout << scientific << setprecision(15);
    
    cout << "Sistema 1:" << endl;
    cout << "Soluzione con PALU: " << x_palu1.transpose() << endl;
    cout << "Errore relativo con PALU: " << relative_error(x_palu1, x_ex) << endl;
    cout << "Soluzione con QR: " << x_qr1.transpose() << endl;
    cout << "Errore relativo con QR: " << relative_error(x_qr1, x_ex) << endl;
    cout << endl;

    cout << "Sistema 2:" << endl;
    cout << "Soluzione con PALU: " << x_palu2.transpose() << endl;
    cout << "Errore relativo con PALU: " << relative_error(x_palu2, x_ex) << endl;
    cout << "Soluzione con QR: " << x_qr2.transpose() << endl;
    cout << "Errore relativo con QR: " << relative_error(x_qr2, x_ex) << endl;
    cout << endl;

    cout << "Sistema 3:" << endl;
    cout << "Soluzione con PALU: " << x_palu3.transpose() << endl;
    cout << "Errore relativo con PALU: " << relative_error(x_palu3, x_ex) << endl;
    cout << "Soluzione con QR: " << x_qr3.transpose() << endl;
    cout << "Errore relativo con QR: " << relative_error(x_qr3, x_ex) << endl;	
	
    return 0;
}
