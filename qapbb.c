/*
 * Copyright (c) 2025-2026 Stefan Teleman.
 *
 * Licensed under the MIT License.
 * See https://opensource.org/license/mit
 * SPDX-License-Identifier: MIT
 *
 */

/*
 *
 * qapbb.c -- C translation of qapbb.f
 *
 * Exact branch-and-bound solver for the Quadratic Assignment Problem.
 *
 *   Method:  perturbation method with Gilmore-Lawler style bounds; each node
 *            bound comes from a linear sum assignment problem (LSAP) solved by
 *            a shortest augmenting path method.
 *   Authors of the original FORTRAN IV code:
 *            T. Boenniger, R. E. Burkard, K.-H. Stratmann  (QAP, ALTKOS,
 *            WEGSPE, PROGNO, SSORT) and U. Derigs (LSAP).
 *
 * The translation is deliberately literal: one C function per FORTRAN
 * subprogram, the original label numbers preserved as C labels (L5210, L2900,
 * ...), and the same variable names, so the two files can be diffed by eye.
 * Arrays are 1-based (element 0 is allocated but unused) to keep every
 * subscript expression identical to the Fortran.
 *
 * Two-dimensional arrays keep Fortran column-major order via the A()/B()/C()/
 * ZUL() macros, so A(i,j) below is A(I,J) above. UMSPEI, ASPEI, BSPEI and
 * CSPEI are linear arrays stored row-wise, exactly as in the original.
 *
 * The single intentional deviation from qapbb.f: the workspace is heap
 * allocated from n using the size formulas documented in the original header
 * comment, instead of being fixed at ndim = 33 with hand-computed literals.
 * There is therefore no built-in size limit. For n <= 33 the output is
 * byte-for-byte identical to the Fortran program's.
 *
 * Build:  cc -O2 -fwrapv -o qapbb qapbb.c
 * Run:    ./qapbb < instance.dat
 */

#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Fortran I<w> edit descriptor: right justified, all asterisks if the */
/* value does not fit. Rotating buffers so several may be live in one  */
/* printf argument list.                                               */
/* ------------------------------------------------------------------ */
static const char *ifield(int w, int val)
{
    static char out[4][64];
    static int k = 0;
    char tmp[64];
    char *s;
    int len, i;

    k = (k + 1) & 3;
    s = out[k];
    len = snprintf(tmp, sizeof tmp, "%d", val);
    if (len > w) {
        for (i = 0; i < w; i++)
            s[i] = '*';
        s[w] = '\0';
    } else {
        snprintf(s, sizeof out[0], "%*s", w, tmp);
    }
    return s;
}

static void *xcalloc(size_t nmemb, size_t size)
{
    void *p = calloc(nmemb, size);
    if (p == NULL) {
        fprintf(stderr, "qapbb: out of memory\n");
        exit(1);
    }
    return p;
}

/* ================================================================== */
/*  SUBROUTINE SSORT                                                  */
/*                                                                    */
/*  Sorts A(1..L) into increasing order by the Shell algorithm and     */
/*  carries B along. B is not required to be initialised by the        */
/*  callers in this program.                                          */
/* ================================================================== */
static void ssort(int *a, int *b, int l)
{
    int f, n2, s, t, ls, i, is, ah, bh, j, js;

    f = 1;
    if (l <= f)
        return;
    n2 = (l - f + 1) / 2;
    s = 1023;
    for (t = 1; t <= 10; t++) {
        if (s > n2)
            goto L90;
        ls = l - s;
        for (i = f; i <= ls; i++) {
            is = i + s;
            ah = a[is];
            bh = b[is];
            j = i;
            js = is;
        L5:
            if (ah >= a[j])
                goto L10;
            a[js] = a[j];
            b[js] = b[j];
            js = j;
            j = j - s;
            if (j >= f)
                goto L5;
        L10:
            a[js] = ah;
            b[js] = bh;
        }
    L90:
        s = s / 2;
    }
}

/* ================================================================== */
/*  SUBROUTINE LSAP  (U. Derigs)                                      */
/*                                                                    */
/*  Linear sum assignment problem, shortest augmenting path method.    */
/*    c      cost matrix, n*n, stored row-wise                        */
/*    sup    large machine number                                     */
/*    *z     optimal value                    (out)                   */
/*    spalte optimal assignment, row -> column (out)                  */
/*    ys,yt  optimal dual variables            (out)                  */
/*  zeile, dminus, dplus, vor and label are workspace of length n.     */
/* ================================================================== */
static void lsap(int n, int sup, const int *c, int *z, int *zeile, int *spalte,
                 int *dminus, int *dplus, int *ys, int *yt, int *vor, int *label)
{
    int i, j, ik, cc, ui, jo, vj, u, us, usi, d, indexv, w, ws, wsi, vgl, ind;
    int is, isj;

    ui = 0;
    jo = 0;
    d = 0;
    indexv = 0;

    /* *** STARTPROCEDURE
       CONSTRUCTION OF AN INITIAL (PARTIAL) ASSIGNMENT */
    for (i = 1; i <= n; i++) {
        zeile[i] = 0;
        spalte[i] = 0;
        vor[i] = 0;
        ys[i] = 0;
        yt[i] = 0;
    }
    ik = 0;
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= n; j++) {
            ik = ik + 1;
            cc = c[ik];
            if (j == 1)
                goto L4;
            if ((cc - ui) >= 0)
                continue;
        L4:
            ui = cc;
            jo = j;
        }
        ys[i] = ui;
        if (zeile[jo] != 0)
            continue;
        zeile[jo] = i;
        spalte[i] = jo;
    }
    for (j = 1; j <= n; j++) {
        yt[j] = 0;
        if (zeile[j] == 0)
            yt[j] = sup;
    }
    ik = 0;
    for (i = 1; i <= n; i++) {
        ui = ys[i];
        for (j = 1; j <= n; j++) {
            ik = ik + 1;
            vj = yt[j];
            if (vj <= 0)
                continue;
            cc = c[ik] - ui;
            if (cc >= vj)
                continue;
            yt[j] = cc;
            vor[j] = i;
        }
    }
    for (j = 1; j <= n; j++) {
        i = vor[j];
        if (i == 0)
            continue;
        if (spalte[i] != 0)
            continue;
        spalte[i] = j;
        zeile[j] = i;
    }
    for (i = 1; i <= n; i++) {
        if (spalte[i] != 0)
            continue;
        ui = ys[i];
        ik = (i - 1) * n;
        for (j = 1; j <= n; j++) {
            ik = ik + 1;
            if (zeile[j] != 0)
                continue;
            cc = c[ik];
            if ((cc - ui - yt[j]) > 0)
                continue;
            spalte[i] = j;
            zeile[j] = i;
            break;
        }
    }

    /* *** CONSTRUCTION OF THE OPTIMAL ASSIGNMENT */
    for (u = 1; u <= n; u++) {
        if (spalte[u] > 0)
            continue;

        /* *** SHORTEST PATH COMPUTATION */
        us = (u - 1) * n;
        for (i = 1; i <= n; i++) {
            vor[i] = u;
            label[i] = 0;
            dplus[i] = sup;
            usi = us + i;
            dminus[i] = c[usi] - ys[u] - yt[i];
        }
        dplus[u] = 0;
    L105:
        d = sup;
        for (i = 1; i <= n; i++) {
            if (label[i])
                continue;
            if (dminus[i] >= d)
                continue;
            d = dminus[i];
            indexv = i;
        }
        if (zeile[indexv] <= 0)
            goto L400;
        label[indexv] = 1;
        w = zeile[indexv];
        ws = (w - 1) * n;
        dplus[w] = d;
        for (i = 1; i <= n; i++) {
            if (label[i])
                continue;
            wsi = ws + i;
            vgl = d + c[wsi] - ys[w] - yt[i];
            if (dminus[i] <= vgl)
                continue;
            dminus[i] = vgl;
            vor[i] = w;
        }
        goto L105;

        /* *** AUGMENTATION */
    L400:
        w = vor[indexv];
        zeile[indexv] = w;
        ind = spalte[w];
        spalte[w] = indexv;
        if (w == u)
            goto L500;
        indexv = ind;
        goto L400;

        /* *** TRANSFORMATION */
    L500:
        for (i = 1; i <= n; i++) {
            if (dplus[i] == sup)
                goto L505;
            ys[i] = ys[i] + d - dplus[i];
        L505:
            if (dminus[i] >= d)
                continue;
            yt[i] = yt[i] + dminus[i] - d;
        }
    }

    /* *** COMPUTATION OF THE OPTIMAL VALUE */
    *z = 0;
    for (i = 1; i <= n; i++) {
        is = (i - 1) * n;
        j = spalte[i];
        isj = is + j;
        *z = *z + c[isj];
    }
}

/* ================================================================== */
/*  SUBROUTINE ALTKOS                                                 */
/*                                                                    */
/*  Determination of the alternative costs and of the resulting        */
/*  single assignment (*izaehl -> *jzaehl) with maximal alternative    */
/*  cost *alterk.                                                     */
/* ================================================================== */
static void altkos(int n, const int *umspei, const int *z1, int unendl,
                   int *izaehl, int *jzaehl, int *alterk)
{
    int jj, i, j, j1, min, i1, a, min1;

    *alterk = -1;
    jj = 0;
    for (i = 1; i <= n; i++) {
        j = z1[i];
        j1 = j - n;
        min = unendl;
        for (i1 = 1; i1 <= n; i1++) {
            j1 = j1 + n;
            if (i1 == i)
                continue;
            a = umspei[j1];
            if (a < min)
                min = a;
        }
        min1 = min;
        min = unendl;
        for (i1 = 1; i1 <= n; i1++) {
            jj = jj + 1;
            if (i1 == j)
                continue;
            a = umspei[jj];
            if (a < min)
                min = a;
        }
        min = min + min1;
        if (min <= *alterk)
            continue;
        *izaehl = i;
        *jzaehl = j;
        *alterk = min;
    }
}

/* ================================================================== */
/*  SUBROUTINE WEGSPE                                                 */
/*                                                                    */
/*  Orders the elements in the rows of A decreasingly and in B         */
/*  increasingly; the resulting matrices are stored row-wise on the    */
/*  vectors aspei and bspei. On levels below the root the rows and     */
/*  columns just fixed are deleted from those vectors in place.        */
/*                                                                    */
/*  vekt is scratch of length n (the caller passes U) and h1 is the    */
/*  companion array handed to ssort (its contents are not used).       */
/* ================================================================== */
static void wegspe(int n, int k, int nmk, int izaehl, int jzaehl,
                   const int *a, const int *b, const int *veksum, int *vekt,
                   const int *boolv, const int *bool1, int *aspei, int *bspei,
                   int *h1, int ld)
{
#define A(i, j) a[((j) - 1) * ld + ((i) - 1)]
#define B(i, j) b[((j) - 1) * ld + ((i) - 1)]

    int nmkm2, j1, i, iz, j, nmkj, nsum1, j2, t, logi;

    nmkm2 = nmk - 1;
    if (nmk != n)
        goto L2900;

    j1 = 1;
    for (i = 1; i <= n; i++) {
        iz = 0;
        for (j = 1; j <= n; j++) {
            if (j == i)
                continue;
            iz = iz + 1;
            vekt[iz] = A(i, j);
        }
        ssort(vekt, h1, nmkm2);
        for (j = 1; j <= nmkm2; j++) {
            nmkj = nmk - j;
            aspei[j1] = vekt[nmkj];
            j1 = j1 + 1;
        }
    }
    j1 = 1;
    for (i = 1; i <= n; i++) {
        iz = 0;
        for (j = 1; j <= n; j++) {
            if (j == i)
                continue;
            iz = iz + 1;
            vekt[iz] = B(i, j);
        }
        ssort(vekt, h1, nmkm2);
        for (j = 1; j <= nmkm2; j++) {
            bspei[j1] = vekt[j];
            j1 = j1 + 1;
        }
    }
    return;

L2900:
    nsum1 = veksum[k + 1];
    j1 = nsum1;
    j2 = nsum1 - (nmk + 1) * nmk;
    for (i = 1; i <= n; i++) {
        if (boolv[i])
            goto L2930;
        t = A(i, izaehl);
        logi = 1;
        for (j = 1; j <= nmk; j++) {
            j2 = j2 + 1;
            if (aspei[j2] == t && logi) {
                logi = 0; /* 2980 */
                continue;
            }
            j1 = j1 + 1;
            aspei[j1] = aspei[j2];
        }
        continue;
    L2930:
        if (i == izaehl)
            j2 = j2 + nmk;
    }
    j1 = nsum1;
    j2 = nsum1 - (nmk + 1) * nmk;
    for (i = 1; i <= n; i++) {
        if (bool1[i])
            goto L2960;
        t = B(i, jzaehl);
        logi = 1;
        for (j = 1; j <= nmk; j++) {
            j2 = j2 + 1;
            if (bspei[j2] == t && logi) {
                logi = 0; /* 2970 */
                continue;
            }
            j1 = j1 + 1;
            bspei[j1] = bspei[j2];
        }
        continue;
    L2960:
        if (i == jzaehl)
            j2 = j2 + nmk;
    }

#undef A
#undef B
}

/* ================================================================== */
/*  SUBROUTINE PROGNO                                                 */
/*                                                                    */
/*  Computation of the final cost matrix umspei for the linear         */
/*  subproblem LSAP on the k-th stage of the decision tree. Going      */
/*  forward (weiter) the minimal scalar products are computed and      */
/*  cached in cspei; backtracking they are read back from cspei.       */
/* ================================================================== */
static void progno(int n, int k, int nmk, int *umspei, const int *veksum,
                   const int *vekqu, int weiter, const int *aspei,
                   const int *bspei, int *cspei)
{
    int ju, nsum2, j1, i, j, j2, nmkm1, j3, t, iz, j2iz, j3iz;

    ju = 0;
    if (nmk != n)
        goto L3600;
    nsum2 = 0;
    j1 = 0;
    if (weiter)
        goto L3680;
    goto L3690;

L3600:
    if (weiter)
        goto L3630;
    j1 = vekqu[k];
L3690:
    for (i = 1; i <= nmk; i++) {
        for (j = 1; j <= nmk; j++) {
            j1 = j1 + 1;
            ju = ju + 1;
            umspei[ju] = cspei[j1];
        }
    }
    return;

L3630:
    nsum2 = veksum[k + 1];
    j1 = vekqu[k + 1];
L3680:
    j2 = nsum2;
    nmkm1 = nmk - 1;
    for (i = 1; i <= nmk; i++) {
        j3 = nsum2;
        for (j = 1; j <= nmk; j++) {
            j1 = j1 + 1;
            ju = ju + 1;
            t = 0;
            for (iz = 1; iz <= nmkm1; iz++) {
                j2iz = j2 + iz;
                j3iz = j3 + iz;
                t = t + aspei[j2iz] * bspei[j3iz];
            }
            j3 = j3 + nmkm1;
            cspei[j1] = t;
            umspei[ju] = t;
        }
        j2 = j2 + nmkm1;
    }
}

/* ================================================================== */
/*  SUBROUTINE QAP  (T. Boenniger, R. E. Burkard, K.-H. Stratmann)    */
/*                                                                    */
/*  Perturbation method for the optimal solution of quadratic          */
/*  assignment problems.                                              */
/*                                                                    */
/*    n         dimension of the problem                              */
/*    a, b      distance and connection matrix; nonnegative integers,  */
/*              destroyed by the procedure                            */
/*    unendl    large machine number                                  */
/*    loesg     optimal assignment i -> loesg[i]        (out)          */
/*    *olwert   in: known upper bound (unendl if none)                 */
/*              out: optimal objective function value                 */
/*  Everything else is workspace; ld is the leading dimension of the   */
/*  two-dimensional arrays.                                           */
/* ================================================================== */
static void qap(int n, int *a, int *b, int unendl, int *loesg, int *olwert_io,
                int *c, int *umspei, int *zul, int *u, int *v, int *dd,
                int *partpe, int *y, int *lab, int *z1, int *h1, int *meng,
                int *phiofm, int *menge, int *veksum, int *vekqu, int *alter,
                int *aspei, int *bspei, int *cspei, int *boolv, int *bool1,
                int *hl1, int ld)
{
#define A(i, j) a[((j) - 1) * ld + ((i) - 1)]
#define B(i, j) b[((j) - 1) * ld + ((i) - 1)]
#define C(i, j) c[((j) - 1) * ld + ((i) - 1)]
#define ZUL(i, j) zul[((j) - 1) * ld + ((i) - 1)]

    int k, nm2, i2, j, j1, i, nmk, j2, zstern, ra, rb, raa, rbb, ca1, bmj, bm;
    int kk, ca, am, ccc, zpart, weiter, iz, ikap, jj, jz, ihalt, jhalt;
    int izaehl, jzaehl, alterk, min, min1, min2, i1, t, t1, t2, ize, lbd;
    int olwert = *olwert_io;

    ihalt = 0;
    jhalt = 0;
    izaehl = 0;
    jzaehl = 0;
    alterk = 0;
    i1 = 0;
    lbd = 0;

    /* *** INITIALIZING OF PARAMETERS */
    k = 0;
    nm2 = n - 2;
    i2 = n + 1;
    j = 0;
    j1 = 0;
    for (i = 1; i <= nm2; i++) {
        nmk = i2 - i;
        j2 = nmk * nmk;
        j = j + j2 - nmk;
        veksum[i] = j;
        j1 = j1 + j2;
        vekqu[i] = j1;
    }

    /* *** 1. REDUCTION: C(I,J) = A(I,I) * B(J,J) */
    for (i = 1; i <= n; i++) {
        boolv[i] = 0;
        bool1[i] = 0;
        zstern = A(i, i);
        for (j = 1; j <= n; j++) {
            ZUL(i, j) = -1;
            /* consider linear term c */
            C(i, j) = zstern * B(j, j) + C(i, j);
        }
    }
    for (j = 1; j <= n; j++) {
        A(j, j) = unendl;
        B(j, j) = unendl;
    }

    /* *** REDUCTION OF MATRICES A AND B */
    for (i = 1; i <= n; i++) {
        ra = A(i, 1);
        rb = B(i, 1);
        for (j = 2; j <= n; j++) {
            raa = A(i, j);
            rbb = B(i, j);
            if (raa < ra)
                ra = raa;
            if (rbb < rb)
                rb = rbb;
        }
        for (j = 1; j <= n; j++) {
            A(i, j) = A(i, j) - ra;
            B(i, j) = B(i, j) - rb;
        }
        u[i] = ra;
        v[i] = rb;
    }

    /* *** DETERMINATION OF THE MATRIX C
       ROWWISE REDUCTION OF MATRICES A AND B */
    for (i = 1; i <= n; i++) {
        ca1 = u[i];
        for (j = 1; j <= n; j++) {
            bmj = v[j];
            bm = (n - 1) * bmj;
            for (kk = 1; kk <= n; kk++) {
                if (kk == j)
                    continue;
                bm = bm + B(j, kk);
            }
            ca = ca1 * bm;
            am = 0;
            for (kk = 1; kk <= n; kk++) {
                if (kk == i)
                    continue;
                am = am + A(i, kk);
            }
            C(i, j) = ca + bmj * am + C(i, j);
        }
    }

    /* *** COLUMNWISE REDUCTION OF MATRICES A AND B */
    for (i = 1; i <= n; i++) {
        ra = A(1, i);
        rb = B(1, i);
        for (j = 2; j <= n; j++) {
            raa = A(j, i);
            rbb = B(j, i);
            if (raa < ra)
                ra = raa;
            if (rbb < rb)
                rb = rbb;
        }
        for (j = 1; j <= n; j++) {
            A(j, i) = A(j, i) - ra;
            B(j, i) = B(j, i) - rb;
        }
        u[i] = ra;
        v[i] = rb;
    }
    for (i = 1; i <= n; i++) {
        A(i, i) = 0;
        B(i, i) = 0;
        ca1 = u[i];
        for (j = 1; j <= n; j++) {
            bmj = v[j];
            bm = (n - 1) * bmj;
            for (kk = 1; kk <= n; kk++) {
                if (kk == j)
                    continue;
                bm = bm + B(kk, j);
            }
            ca = ca1 * bm;
            am = 0;
            for (kk = 1; kk <= n; kk++) {
                if (kk == i)
                    continue;
                am = am + A(kk, i);
            }
            ccc = C(i, j) + ca + bmj * am;
            C(i, j) = ccc;
        }
    }
    zpart = 0;
    nmk = n;
    weiter = 1;

    /* *** IMPROVEMENT OF THE BOUNDS C(I,J) BY ADDING MINIMAL SCALAR
       PRODUCTS (GILMORE BOUNDS). THE SUBROUTINES WEGSPE AND PROGNO
       COMPUTE THESE MINIMAL SCALAR PRODUCTS. */
    wegspe(n, k, nmk, izaehl, jzaehl, a, b, veksum, u, boolv, bool1, aspei,
           bspei, h1, ld);
    progno(n, k, nmk, umspei, veksum, vekqu, weiter, aspei, bspei, cspei);
    weiter = 0;

L5210:
    iz = 0;
    i2 = 0;
    j2 = 0;
    ikap = 0;
    for (i = 1; i <= n; i++) {
        if (boolv[i])
            continue;
        jj = iz * nmk;
        iz = iz + 1;
        jz = 0;
        for (j = 1; j <= n; j++) {
            if (bool1[j])
                continue;
            jz = jz + 1;
            jj = jj + 1;
            if (ZUL(i, j) < 0) {
                umspei[jj] = C(i, j) + umspei[jj]; /* 5145 */
                continue;
            }
            umspei[jj] = unendl;
            ikap = ikap + 1;
            if (ikap - 2 < 0) { /* 5375 */
                ihalt = iz;
                jhalt = jz;
                continue;
            }
            if (ikap - 2 == 0) { /* 5380 */
                if (iz == ihalt)
                    i2 = iz;
                if (jz == jhalt)
                    j2 = jz;
                continue;
            }
            continue; /* 5125 */
        }
    }

    /* *** COMPUTATION OF A BOUND BY SOLVING LSAP WITH COSTMATRIX UMSPEI */
L5180:
    lsap(nmk, unendl, umspei, &zstern, y, z1, lab, dd, u, v, h1, hl1);
    jj = 0;
    for (i = 1; i <= nmk; i++) {
        for (j = 1; j <= nmk; j++) {
            jj = jj + 1;
            umspei[jj] = umspei[jj] - u[i] - v[j];
        }
    }

    /* *** ZPART IS THE FIXED FRACTION OF THE OBJECTIVE FUNCTION VALUE
       IMPLIED BY THE PRESENT PARTIAL PERMUTATION.
       THE PRESENT BOUND IS ZPART+ZSTERN. */
    if (zpart + zstern >= olwert)
        goto L5250;
    if (lbd < zpart + zstern) {
        lbd = zpart + zstern;
        /* printf("   lb=%s\n", ifield(12, lbd)); */
    }
    if (weiter)
        goto L5220;

L5135:
    if (ikap - 1 < 0)
        goto L5400;
    if (ikap - 1 == 0)
        goto L5410;
    goto L5420;

    /* *** COMPUTATION OF THE ALTERNATIVE COSTS */
L5400:
    altkos(nmk, umspei, z1, unendl, &izaehl, &jzaehl, &alterk);
    goto L5490;

    /* *** COMPUTATION OF THE NEXT SINGLE ASSIGNMENT IN A FIXED ROW
       OR COLUMN */
L5410:
    min = unendl;
    j1 = z1[ihalt];
    jj = (ihalt - 1) * nmk;
    for (j = 1; j <= nmk; j++) {
        jj = jj + 1;
        t = umspei[jj];
        if (min <= t || j == j1)
            continue;
        min = t;
    }
    min1 = min;
    min = unendl;
    jj = j1;
    for (i = 1; i <= nmk; i++) {
        t = umspei[jj];
        jj = jj + nmk;
        if (min <= t || i == ihalt)
            continue;
        min = t;
    }
    min1 = min1 + min;
    min = unendl;
    jj = jhalt;
    for (i = 1; i <= nmk; i++) {
        t = umspei[jj];
        jj = jj + nmk;
        if (t >= min || jhalt == z1[i])
            continue;
        min = t;
    }
    min2 = min;
    for (i = 1; i <= nmk; i++) {
        if (z1[i] == jhalt)
            break;
    }
    i1 = i; /* 5540 */
    jj = (i1 - 1) * nmk;
    min = unendl;
    for (j = 1; j <= nmk; j++) {
        jj = jj + 1;
        t = umspei[jj];
        if (t >= min || j == jhalt)
            continue;
        min = t;
    }
    if ((min + min2) < min1)
        goto L5450;
    izaehl = i1;
    jzaehl = jhalt;
    alterk = min + min2;
    goto L5490;
L5450:
    izaehl = ihalt;
    jzaehl = j1;
    alterk = min1;
    goto L5490;

    /* *** COMPUTATION OF THE NEXT SINGLE ASSIGNMENT IN THE PREVIOUSLY
       FIXED ROW (RESP. PREVIOUSLY FIXED COLUMN) */
L5420:
    if (i2 == 0)
        goto L5480;
    izaehl = i2;
    jzaehl = z1[izaehl];
    goto L5370;
L5480:
    jzaehl = j2;
    for (i = 1; i <= nmk; i++) {
        if (z1[i] == jzaehl)
            break;
    }
    izaehl = i; /* 5485 */
L5370:
    min = unendl;
    jj = (izaehl - 1) * nmk;
    for (i = 1; i <= nmk; i++) {
        jj = jj + 1;
        t = umspei[jj];
        if (t >= min || i == jzaehl)
            continue;
        min = t;
    }
    alterk = min;
    min = unendl;
    jj = jzaehl;
    for (j = 1; j <= nmk; j++) {
        t = umspei[jj];
        jj = jj + nmk;
        if (t >= min || j == izaehl)
            continue;
        min = t;
    }
    alterk = alterk + min;

L5490:
    iz = 0;
    alter[k + 1] = alterk + zstern;
    for (i = 1; i <= n; i++) {
        if (boolv[i])
            continue;
        iz = iz + 1;
        if (izaehl == iz)
            break;
    }
    izaehl = i; /* 5165 */
    iz = 0;
    for (j = 1; j <= n; j++) {
        if (bool1[j])
            continue;
        iz = iz + 1;
        if (jzaehl == iz)
            break;
    }
    jzaehl = j; /* 5185 */
    ZUL(izaehl, jzaehl) = k;
    weiter = 1;
    boolv[izaehl] = 1;
    bool1[jzaehl] = 1;
    nmk = n - k - 1;

    /* *** COMPUTATION OF THE COST MATRIX CORRESPONDING TO THE NEW PARTIAL
       PERMUTATION */
    wegspe(n, k, nmk, izaehl, jzaehl, a, b, veksum, u, boolv, bool1, aspei,
           bspei, h1, ld);
    progno(n, k, nmk, umspei, veksum, vekqu, weiter, aspei, bspei, cspei);

    iz = 0;
    for (i = 1; i <= n; i++) {
        if (boolv[i])
            continue;
        t1 = A(i, izaehl);
        t2 = A(izaehl, i);
        jz = iz;
        for (j = 1; j <= n; j++) {
            if (bool1[j])
                continue;
            jz = jz + 1;
            umspei[jz] = C(i, j) + t1 * B(j, jzaehl) + t2 * B(jzaehl, j)
                       + umspei[jz];
        }
        iz = iz + nmk;
    }
    zpart = zpart + C(izaehl, jzaehl);
    goto L5180;

    /* *** THE BOUND FOR THE NEW PARTIAL PERMUTATION IS NOT LESS THAN A
       PREVIOUSLY FOUND OBJECTIVE FUNCTION VALUE.  - BACKTRACKING - */
L5250:
    if (!weiter)
        goto L5255;
    weiter = 0;
    k = k + 1;
    goto L5230;

    /* *** EXIT: THE SOLUTION TREE IS COMPLETELY FATHOMED. */
L5255:
    if (k == 0)
        goto L5600;
    izaehl = menge[k];
    jzaehl = phiofm[k];

L5220:
    for (i = 1; i <= n; i++) {
        if (boolv[i])
            continue;
        t1 = A(izaehl, i);
        t2 = A(i, izaehl);
        for (j = 1; j <= n; j++) {
            if (bool1[j])
                continue;
            t = t1 * B(jzaehl, j) + t2 * B(j, jzaehl);
            if (!weiter)
                t = -t;
            C(i, j) = C(i, j) + t;
        }
    }
    if (!weiter)
        goto L5230;
    partpe[izaehl] = jzaehl;
    k = k + 1;
    menge[k] = izaehl;
    phiofm[k] = jzaehl;
    if (k == (n - 2))
        goto L5270;
    ikap = 0;
    goto L5135;

    /* *** CANCELLATION OF THE LAST SINGLE ASSIGNMENT */
L5230:
    for (i = 1; i <= n; i++) {
        if (boolv[i])
            continue;
        for (j = 1; j <= n; j++) {
            if (!bool1[j] && ZUL(i, j) == k)
                ZUL(i, j) = -1;
        }
    }
    zpart = zpart - C(izaehl, jzaehl);
    boolv[izaehl] = 0;
    bool1[jzaehl] = 0;
    k = k - 1;
    nmk = n - k;
    if (alter[k + 1] + zpart >= olwert)
        goto L5255;
    progno(n, k, nmk, umspei, veksum, vekqu, weiter, aspei, bspei, cspei);
    goto L5210;

    /* *** COMPUTATION OF THE OBJECTIVE FUNCTION VALUES FOR THE REMAINING
       TWO COMPLETE PERMUTATIONS */
L5270:
    for (i = 1; i <= n; i++) {
        if (boolv[i])
            continue;
        iz = i;
        break;
    }
    for (i = 1; i <= n; i++) { /* 5285 */
        if (bool1[i])
            continue;
        j = i;
        break;
    }
    /* 5295 */
    boolv[iz] = 1;
    bool1[j] = 1;
    for (i = 1; i <= n; i++) {
        if (boolv[i])
            continue;
        i1 = i;
        break;
    }
    for (i = 1; i <= n; i++) { /* 5305 */
        if (bool1[i])
            continue;
        j1 = i;
        break;
    }
    /* 5315 */
    weiter = 0;
    ize = 0;
L5330:
    zstern = C(iz, j) + C(i1, j1) + A(iz, i1) * B(j, j1) + A(i1, iz) * B(j1, j);
    boolv[iz] = 0;
    bool1[j] = 0;
    if ((zstern + zpart) >= olwert)
        goto L5340;
    olwert = zstern + zpart;
    /* printf("                    new solution=%s\n", ifield(12, olwert)); */
    lbd = 0;
    for (i = 1; i <= n; i++) {
        if (boolv[i])
            loesg[i] = partpe[i];
    }
    for (i = 1; i <= nm2; i++)
        meng[i] = menge[i];
    loesg[iz] = j;
    loesg[i1] = j1;
L5340:
    if (ize != 0)
        goto L5255;
    ize = iz;
    iz = i1;
    i1 = ize;
    goto L5330;

L5600:
    *olwert_io = olwert;

#undef A
#undef B
#undef C
#undef ZUL
}

/* ================================================================== */
/*  SUBROUTINE ILESE                                                  */
/*                                                                    */
/*  Reads n, then the n*n matrix a row by row, then b, then the        */
/*  optional linear term c. If c is absent (end of file) it is zeroed. */
/*                                                                    */
/*  The Fortran original uses list-directed READ, one statement per    */
/*  matrix row; this reads a plain whitespace-separated token stream,  */
/*  which accepts the same files (a row may be wrapped over several    */
/*  lines) and in addition tolerates several rows sharing a line.      */
/* ================================================================== */
static int ilese(int **ap, int **bp, int **cp)
{
    int n, i, j, ld;
    int *a, *b, *c;

    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "qapbb: cannot read n\n");
        exit(1);
    }
    if (n < 1) {
        fprintf(stderr, "qapbb: n must be at least 1 (read %d)\n", n);
        exit(1);
    }
    ld = n;
    a = xcalloc((size_t)n * n, sizeof *a);
    b = xcalloc((size_t)n * n, sizeof *b);
    c = xcalloc((size_t)n * n, sizeof *c);

    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            if (scanf("%d", &a[(j - 1) * ld + (i - 1)]) != 1) {
                fprintf(stderr, "qapbb: matrix a is incomplete\n");
                exit(1);
            }
    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            if (scanf("%d", &b[(j - 1) * ld + (i - 1)]) != 1) {
                fprintf(stderr, "qapbb: matrix b is incomplete\n");
                exit(1);
            }
    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            if (scanf("%d", &c[(j - 1) * ld + (i - 1)]) != 1) {
                printf(" no linear term.\n");
                for (i = 0; i < n * n; i++)
                    c[i] = 0;
                goto done;
            }
done:
    *ap = a;
    *bp = b;
    *cp = c;
    return n;
}

/* ------------------------------------------------------------------ */
/*  Output, reproducing FORMAT statements 2000, 2005 and 2010.         */
/* ------------------------------------------------------------------ */
static void report(int n, const int *loesg, int olwert)
{
    int i;

    printf("\n\n\n         EXACT SOLUTION:\n");
    for (i = 1; i <= n; i++)
        printf("         %s -->%s\n", ifield(5, i), ifield(2, loesg[i]));
    printf("\n         OBJECTIVE FUNCTION VALUE:%s\n", ifield(6, olwert));
}

/* ================================================================== */
/*  PROGRAM QAPBB                                                     */
/* ================================================================== */
int main(void)
{
    int unendl = 1000000000;
    int n, olwert;
    int *a, *b, *c;
    int *umspei, *zul, *loesg, *u, *v, *dd, *partpe, *y, *lab, *z1, *h1;
    int *meng, *phiofm, *menge, *veksum, *vekqu, *alter;
    int *aspei, *bspei, *cspei, *boolv, *bool1, *hl1;
    size_t nab, ncc, nvec;

    n = ilese(&a, &b, &c);

    /* The branch-and-bound proper needs n >= 3: for n < 3 the vectors veksum
       and vekqu are empty (nm2 <= 0) and the original Fortran runs off the
       ends of aspei/cspei, where it dies with a segmentation fault. Those two
       cases have at most two permutations, so evaluate them directly. */
    if (n <= 2) {
        static const int perms[2][3] = { { 0, 1, 2 }, { 0, 2, 1 } };
        int bestperm[3], best = 0, have = 0, t, np, val, ii, jj;

        np = (n == 1) ? 1 : 2;
        for (t = 0; t < np; t++) {
            val = 0;
            for (ii = 1; ii <= n; ii++) {
                val += c[(perms[t][ii] - 1) * n + (ii - 1)];
                for (jj = 1; jj <= n; jj++)
                    val += a[(jj - 1) * n + (ii - 1)]
                         * b[(perms[t][jj] - 1) * n + (perms[t][ii] - 1)];
            }
            if (!have || val < best) {
                best = val;
                have = 1;
                for (ii = 1; ii <= n; ii++)
                    bestperm[ii] = perms[t][ii];
            }
        }
        report(n, bestperm, best);
        free(a);
        free(b);
        free(c);
        return 0;
    }

    /* Workspace sizes as documented in the header of SUBROUTINE QAP.
       aspei/bspei need n*(n+1)*(2n-2)/6 and cspei n*(n+1)*(2n+1)/6 - 1;
       the vectors of length n-2 are allocated with n+2 slots so that the
       degenerate small-n cases cannot run off the end. All arrays are
       1-based, hence the extra leading slot. */
    nab = (size_t)n * (n + 1) * (2 * n - 2) / 6 + 1;
    ncc = (size_t)n * (n + 1) * (2 * n + 1) / 6;
    nvec = (size_t)n + 3;

    umspei = xcalloc((size_t)n * n + 1, sizeof *umspei);
    zul    = xcalloc((size_t)n * n, sizeof *zul);
    loesg  = xcalloc(nvec, sizeof *loesg);
    u      = xcalloc(nvec, sizeof *u);
    v      = xcalloc(nvec, sizeof *v);
    dd     = xcalloc(nvec, sizeof *dd);
    partpe = xcalloc(nvec, sizeof *partpe);
    y      = xcalloc(nvec, sizeof *y);
    lab    = xcalloc(nvec, sizeof *lab);
    z1     = xcalloc(nvec, sizeof *z1);
    h1     = xcalloc(nvec, sizeof *h1);
    meng   = xcalloc(nvec, sizeof *meng);
    phiofm = xcalloc(nvec, sizeof *phiofm);
    menge  = xcalloc(nvec, sizeof *menge);
    veksum = xcalloc(nvec, sizeof *veksum);
    vekqu  = xcalloc(nvec, sizeof *vekqu);
    alter  = xcalloc(nvec, sizeof *alter);
    boolv  = xcalloc(nvec, sizeof *boolv);
    bool1  = xcalloc(nvec, sizeof *bool1);
    hl1    = xcalloc(nvec, sizeof *hl1);
    aspei  = xcalloc(nab, sizeof *aspei);
    bspei  = xcalloc(nab, sizeof *bspei);
    cspei  = xcalloc(ncc, sizeof *cspei);

    /* 0 = no feasible solution known, else a known objective value */
    olwert = 0;
    if (olwert == 0)
        olwert = unendl;

    qap(n, a, b, unendl, loesg, &olwert, c, umspei, zul, u, v, dd, partpe, y,
        lab, z1, h1, meng, phiofm, menge, veksum, vekqu, alter, aspei, bspei,
        cspei, boolv, bool1, hl1, n);

    report(n, loesg, olwert);

    free(a); free(b); free(c); free(umspei); free(zul); free(loesg);
    free(u); free(v); free(dd); free(partpe); free(y); free(lab); free(z1);
    free(h1); free(meng); free(phiofm); free(menge); free(veksum);
    free(vekqu); free(alter); free(boolv); free(bool1); free(hl1);
    free(aspei); free(bspei); free(cspei);

    return 0;
}

