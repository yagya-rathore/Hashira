import org.json.JSONObject;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

public class PolynomialSolver {

    // Convert number string in base to decimal BigInteger
    public static BigInteger baseToDecimal(String value, int base) {
        return new BigInteger(value, base);
    }

    // Solve linear system Ax = b using Gaussian elimination
    // A is n x n matrix, b is n vector
    // Returns solution vector x
    public static double[] gaussianElimination(double[][] A, double[] b) {
        int n = b.length;

        for (int p = 0; p < n; p++) {
            // Find pivot row and swap
            int max = p;
            for (int i = p + 1; i < n; i++) {
                if (Math.abs(A[i][p]) > Math.abs(A[max][p])) {
                    max = i;
                }
            }
            double[] temp = A[p];
            A[p] = A[max];
            A[max] = temp;

            double t = b[p];
            b[p] = b[max];
            b[max] = t;

            // Pivot within A and b
            if (Math.abs(A[p][p]) <= 1e-10) {
                throw new ArithmeticException("Matrix is singular or nearly singular");
            }

            for (int i = p + 1; i < n; i++) {
                double alpha = A[i][p] / A[p][p];
                b[i] -= alpha * b[p];
                for (int j = p; j < n; j++) {
                    A[i][j] -= alpha * A[p][j];
                }
            }
        }

        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i][i];
        }

        return x;
    }

    public static double[] solvePolynomialCoefficients(String jsonInput) {
        JSONObject data = new JSONObject(jsonInput);

        int n = data.getJSONObject("keys").getInt("n");
        int k = data.getJSONObject("keys").getInt("k");
        int m = k - 1;  // degree of polynomial

        List<BigInteger> roots = new ArrayList<>();

        for (int i = 1; i <= n; i++) {
            if (!data.has(String.valueOf(i))) {
                continue;
            }
            JSONObject rootInfo = data.getJSONObject(String.valueOf(i));
            int base = Integer.parseInt(rootInfo.getString("base"));
            String value = rootInfo.getString("value");

            BigInteger rootDecimal = baseToDecimal(value, base);
            roots.add(rootDecimal);
        }

        // Use only first k roots
        roots = roots.subList(0, Math.min(k, roots.size()));

        // Build Vandermonde matrix A and vector b for Ax = b
        // We fix leading coefficient a_m = 1
        // So system is A_reduced * x = b with (k x (k-1)) matrix A_reduced and vector b

        int size = roots.size();
        double[][] A_reduced = new double[size][size - 1];
        double[] b_vec = new double[size];

        for (int i = 0; i < size; i++) {
            BigInteger r = roots.get(i);
            // Fill row i with powers r^0 ... r^(m-1)
            for (int j = 0; j < size - 1; j++) {
                A_reduced[i][j] = Math.pow(r.doubleValue(), j);
            }
            // b = -r^m
            b_vec[i] = -Math.pow(r.doubleValue(), size);
        }

        double[] coeffsExceptLeading = gaussianElimination(A_reduced, b_vec);

        // Compose full coefficients (a0 ... a_{m-1}, leading coefficient 1)
        double[] coeffs = new double[size];
        for (int i = 0; i < size - 1; i++) {
            coeffs[i] = coeffsExceptLeading[i];
        }
        coeffs[size - 1] = 1.0;

        return coeffs;
    }

    public static void main(String[] args) {
        String jsonInput = "{\n" +
                "\"keys\": {\n" +
                "    \"n\": 4,\n" +
                "    \"k\": 3\n" +
                "},\n" +
                "\"1\": {\n" +
                "    \"base\": \"10\",\n" +
                "    \"value\": \"4\"\n" +
                "},\n" +
                "\"2\": {\n" +
                "    \"base\": \"2\",\n" +
                "    \"value\": \"111\"\n" +
                "},\n" +
                "\"3\": {\n" +
                "    \"base\": \"10\",\n" +
                "    \"value\": \"12\"\n" +
                "},\n" +
                "\"6\": {\n" +
                "    \"base\": \"4\",\n" +
                "    \"value\": \"213\"\n" +
                "}\n" +
                "}";

        double[] coefficients = solvePolynomialCoefficients(jsonInput);

        System.out.println("Polynomial coefficients (a0 to am):");
        for (double c : coefficients) {
            System.out.printf("%.6f ", c);
        }
    }
}
