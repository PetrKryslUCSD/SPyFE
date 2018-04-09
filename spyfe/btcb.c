#define WBMAT_NNODES 8

void
btcb(double K[WBMAT_NNODES*3][WBMAT_NNODES*3],
            double b_l[6][WBMAT_NNODES*3], double C[6][6])
{
  int i, j, k, m;
  const int kdim = WBMAT_NNODES*3;
  double c, blkic;
  for (k=0; k<6; k++) {
    for (m=0; m<6; m++) {
      c = C[k][m];
      if (c != 0) {
        c *= fact;
        for (i=0;  i<kdim; i++) {
          blkic = c * b_l[k][i];
          for (j=0;  j<kdim; j++) {
            K[i][j] += blkic * b_l[m][j];
          }
        }
      }
    }
  }
}

int main()
{
	const int kdim = WBMAT_NNODES * 3;
	double K[WBMAT_NNODES * 3][WBMAT_NNODES * 3], b_l[6][WBMAT_NNODES * 3], C[6][6];
	for (int i = 0; i < kdim; i++) {
		for (int j = 0; j < kdim; j++) {
			K[i][j] = 0.0;
		}
	}
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			C[i][j] = 1.0;
		}
	}
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < kdim; j++) {
			b_l[i][j] = (double)i+j;
		}
	}
	for (int r = 0; r < 100000; r++) {
		btcb(K, b_l, C);
	}
}