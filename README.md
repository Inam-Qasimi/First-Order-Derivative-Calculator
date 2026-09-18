# Jet Numbers — Implementation Instructions

## 1. What You Already Have and Why It's Limited

### 1.1 Your Current Dual Number — A Recap

Your `Dual` struct stores exactly two values: a real part and one infinitesimal
part. A dual number looks like `a + bε` where `ε² = 0`. When you feed
`f(x + ε, y, z)` through your overloaded operators, the chain rule propagates
automatically through every operation, and the infinitesimal coefficient at the
end is exactly `∂f/∂x`.

**Worked example with your code:**
Suppose `f(x, y, z) = sin(x) * cos(y) * tan(z)` and you want `∂f/∂x` at
`(2, 3, 4)`. Your code does:

```cpp
Dual x = Dual(2);        // x = 2 + 0ε
Dual y = Dual(3);        // y = 3 + 0ε
Dual z = Dual(4);        // z = 4 + 0ε
Dual epsilon = Dual(0,1);// ε = 0 + 1ε
Dual dx = f(x + epsilon, y, z);  // f(2 + ε, 3, 4)
```

Internally, `x + epsilon = Dual(2, 1)`. Then `sin(Dual(2, 1))` computes:
- Real part: `sin(2) ≈ 0.9093`
- Infinitesimal part: `1 · cos(2) ≈ -0.4161` (this is `d/dx sin(x)` at `x = 2`)

But `cos(Dual(3, 0))` has infinitesimal part `0`, so when you multiply
`sin(x+ε) * cos(y)`, the product rule gives:
- Infinitesimal: `(-0.4161)(cos(3)) + (sin(2))(0) = (-0.4161)(-0.9900) ≈ 0.4119`

The real part of `dx` is `f(2,3,4)` and the infinitesimal part is `∂f/∂x`.
To also get `∂f/∂y` and `∂f/∂z`, you must call `f` two more times with `ε`
added to `y` and `z` respectively.

### 1.2 The Two Limitations

**Limitation 1 — One variable at a time:** Each call to `f` only perturbs one
variable. For `n` variables you need `n` separate function evaluations. This is
fine for small `n`, but wasteful — it means `n` full traversals of every
operation in `f`.

**Limitation 2 — First-order only:** Because `ε² = 0`, all second-order
information is annihilated. Consider `f(x) = x²`:

```
f(x + ε) = (x + ε)² = x² + 2xε + ε²
                                    ^^^ = 0 by definition
         = x² + 2xε
```

You get `f'(x) = 2x` from the ε-coefficient, which is correct. But `f''(x) = 2`
is completely lost — the `ε²` term that would have carried it was zeroed out.
There is no way to recover second derivatives from a standard dual number.

### 1.3 Your HyperDualNumbers Sketch — A Step in the Right Direction

Your `HyperDualNumbers.cpp` introduces two independent infinitesimals `ε₁` and
`ε₂` with the rules:
- `ε₁² = 0`, `ε₂² = 0`
- `ε₁ε₂ ≠ 0` (this is the key — the *cross term* survives)

A hyper-dual number is `a + bε₁ + cε₂ + dε₁ε₂`. The product rule you wrote:

```
u * v = al + (am + bl)ε₁ + (an + cl)ε₂ + (ao + bn + cm + dl)ε₁ε₂
```

This captures the mixed partial `∂²f/∂x∂y` in the `ε₁ε₂` coefficient. But it
hard-codes exactly 2 variables and cannot compute `∂²f/∂x²` (because `ε₁² = 0`),
nor can it handle 3+ variables without adding more components by hand.

### 1.4 What Jet Numbers Solve

Jet numbers generalize everything above. A `Jet<vars, order>` number:
- Carries **all** variables simultaneously (no repeated function calls)
- Preserves terms up to **any specified order** (not just first)
- Captures **mixed partials** of all combinations
- Is parameterized by `vars` (number of variables) and `order` (max derivative order)

Your dual number is `Jet<1, 1>`. Your hyper-dual is roughly `Jet<2, 2>` but with
the diagonal second derivatives missing. A proper `Jet<2, 2>` also keeps `ε₁²`
and `ε₂²` terms (with the convention that they're nilpotent at order 3, not 2).

---

## 2. The Core Idea — Truncated Taylor Polynomials via Multi-Indices

### 2.1 The Multivariate Taylor Expansion — Where Jets Come From

Recall the single-variable Taylor expansion of `f(x)` around a point `a`:

```
f(a + h) = f(a) + f'(a)·h + f''(a)·h²/2! + f'''(a)·h³/3! + ...
```

For **two** variables, the Taylor expansion of `f(x, y)` around `(a, b)` is:

```
f(a + h, b + k) = f(a,b)
                + [∂f/∂x · h + ∂f/∂y · k]
                + [∂²f/∂x² · h²/2! + ∂²f/∂x∂y · hk/(1!1!) + ∂²f/∂y² · k²/2!]
                + [third order terms ...]
                + ...
```

Notice how each term corresponds to a **multi-index** — a pair of exponents
`(α₁, α₂)` telling you the power of `h` and `k`:

| Term | Multi-index α | Factor | Coefficient |
|------|--------------|--------|-------------|
| `f(a,b)` | (0, 0) | 1 | f(a,b) |
| `∂f/∂x · h` | (1, 0) | h | ∂f/∂x |
| `∂f/∂y · k` | (0, 1) | k | ∂f/∂y |
| `∂²f/∂x² · h²/2!` | (2, 0) | h² | (1/2!) ∂²f/∂x² |
| `∂²f/∂x∂y · hk` | (1, 1) | hk | ∂²f/∂x∂y |
| `∂²f/∂y² · k²/2!` | (0, 2) | k² | (1/2!) ∂²f/∂y² |

A Jet number replaces the concrete perturbations `h` and `k` with **formal
symbols** `ε₁` and `ε₂`, and **truncates** the series at a chosen maximum order.

### 2.2 Multi-Index Notation

A **Jet number** `Jet<vars, order>` represents a truncated multivariate Taylor
polynomial:

```
J = Σ  c_α · εᵅ     (sum over all multi-indices α with |α| ≤ order)
```

Where:

| Symbol | Meaning |
|--------|---------|
| `α = (α₁, α₂, …, αₙ)` | A multi-index — a tuple of non-negative integers, one per variable |
| `|α| = α₁ + α₂ + … + αₙ` | The **total order** of the multi-index |
| `εᵅ = ε₁^α₁ · ε₂^α₂ · … · εₙ^αₙ` | A formal monomial in the infinitesimal symbols |
| `c_α` | The coefficient stored for that monomial |
| `α! = α₁! · α₂! · … · αₙ!` | The multi-index factorial |

**Concrete example — `Jet<2, 2>` (2 variables, order 2):**

A Jet value looks like:
```
J = c₍₀₀₎ + c₍₁₀₎·ε₁ + c₍₀₁₎·ε₂ + c₍₂₀₎·ε₁² + c₍₁₁₎·ε₁ε₂ + c₍₀₂₎·ε₂²
```
That's 6 coefficients. Each one corresponds to one multi-index.

### 2.3 The Truncation Rule

The key truncation rule is: **any monomial with `|α| > order` is discarded
(treated as zero).** This is exactly what makes `ε² = 0` in ordinary dual
numbers — that's just the special case `vars = 1, order = 1` where the only
multi-indices are `(0)` and `(1)`, and `(2)` is truncated.

For `Jet<2, 2>`, the truncation discards any monomial where `α₁ + α₂ > 2`:
- `ε₁³` has multi-index `(3, 0)`, total order 3 → **truncated**
- `ε₁²ε₂` has multi-index `(2, 1)`, total order 3 → **truncated**
- `ε₁ε₂` has multi-index `(1, 1)`, total order 2 → **kept**

This is different from your hyper-dual numbers where `ε₁² = 0` and `ε₂² = 0`
but `ε₁ε₂ ≠ 0`. In Jets, *all* monomials of order ≤ `order` survive, including
the diagonal ones like `ε₁²`.

### 2.4 The Fundamental Relationship — Why This Works

After evaluating `f` on Jet-seeded variables, the coefficient `c_α` satisfies:

```
c_α = (1 / α!) · (∂^|α| f) / (∂x₁^α₁ · ∂x₂^α₂ · … · ∂xₙ^αₙ)
```

where `α! = α₁! · α₂! · … · αₙ!`. So to recover the actual derivative, multiply
`c_α` by `α!`.

**Why does this formula hold?** It comes directly from the multivariate Taylor
theorem. When you seed `xᵢ = aᵢ + εᵢ`, you're effectively computing
`f(a₁ + ε₁, a₂ + ε₂, …)`. The Taylor theorem tells you that the coefficient
of `ε₁^α₁ · ε₂^α₂ · …` in this expansion is exactly `(1/α!) · ∂^α f` evaluated
at `(a₁, a₂, …)`. The Jet arithmetic preserves these coefficients through every
operation because the operator overloads encode the generalized product rule,
chain rule, etc.

**Worked example:** Let `f(x, y) = x²y` and evaluate at `(3, 2)` with a
`Jet<2, 2>`:

```
x = 3 + 1·ε₁ + 0·ε₂ + 0·ε₁² + 0·ε₁ε₂ + 0·ε₂²
y = 2 + 0·ε₁ + 1·ε₂ + 0·ε₁² + 0·ε₁ε₂ + 0·ε₂²

x² = (3 + ε₁)² = 9 + 6ε₁ + ε₁²   (keeping ε₁² since order ≥ 2)

x²·y = (9 + 6ε₁ + ε₁²)(2 + ε₂)
     = 18 + 12ε₁ + 2ε₁² + 9ε₂ + 6ε₁ε₂ + ε₁²ε₂
                                              ^^^ |α| = 3 > 2, truncated!
     = 18 + 12ε₁ + 9ε₂ + 2ε₁² + 6ε₁ε₂ + 0·ε₂²
```

Reading off the coefficients and recovering derivatives:

| Multi-index | Stored coeff | × α! | = Derivative | Hand-check |
|------------|-------------|------|-------------|------------|
| (0,0) | 18 | ×1 | f = 18 | 3²·2 = 18 ✓ |
| (1,0) | 12 | ×1 | ∂f/∂x = 12 | 2xy = 2·3·2 = 12 ✓ |
| (0,1) | 9 | ×1 | ∂f/∂y = 9 | x² = 9 ✓ |
| (2,0) | 2 | ×2! = ×2 | ∂²f/∂x² = 4 | 2y = 4 ✓ |
| (1,1) | 6 | ×1!·1! = ×1 | ∂²f/∂x∂y = 6 | 2x = 6 ✓ |
| (0,2) | 0 | ×2! = ×2 | ∂²f/∂y² = 0 | 0 ✓ |

---

## 3. Data Layout — Enumerating Multi-Indices

### 3.1 How Many Terms?

The number of multi-indices `α` with `|α| ≤ order` for `vars` variables is
given by the "stars and bars" combinatorial formula:

```
N = C(vars + order, order)       (binomial coefficient)
```

**Why this formula?** You're counting the number of ways to distribute up to
`order` total among `vars` slots. This is equivalent to choosing `order` items
from `vars + order` positions (a standard combinatorics result).

**Derivation by example:** For `vars = 2, order = 2`, you need all `(α₁, α₂)`
with `α₁ + α₂ ≤ 2`:
- Order 0: `(0,0)` → 1 tuple
- Order 1: `(1,0), (0,1)` → 2 tuples
- Order 2: `(2,0), (1,1), (0,2)` → 3 tuples
- Total: 1 + 2 + 3 = 6 = C(2+2, 2) = C(4,2) = 6 ✓

Here is a larger reference table:

| vars | order | N | All monomials listed |
|------|-------|---|---------------------|
| 1 | 1 | 2 | `1, ε₁` |
| 1 | 2 | 3 | `1, ε₁, ε₁²` |
| 1 | 3 | 4 | `1, ε₁, ε₁², ε₁³` |
| 2 | 1 | 3 | `1, ε₁, ε₂` |
| 2 | 2 | 6 | `1, ε₁, ε₂, ε₁², ε₁ε₂, ε₂²` |
| 2 | 3 | 10 | `1, ε₁, ε₂, ε₁², ε₁ε₂, ε₂², ε₁³, ε₁²ε₂, ε₁ε₂², ε₂³` |
| 3 | 1 | 4 | `1, ε₁, ε₂, ε₃` |
| 3 | 2 | 10 | `1, ε₁, ε₂, ε₃, ε₁², ε₁ε₂, ε₁ε₃, ε₂², ε₂ε₃, ε₃²` |
| 3 | 3 | 20 | (too many to list) |
| 4 | 2 | 15 | (too many to list) |
| 5 | 3 | 56 | (too many to list) |

Notice how N grows **combinatorially**. For `vars = 10, order = 4`, you'd have
`C(14, 4) = 1001` coefficients per Jet value. This is why Jets are practical
for small-to-moderate variable counts and orders, but become expensive for
large problems.

### 3.2 Building the Multi-Index Table — Algorithm

At compile time (or once at startup), enumerate all multi-indices
`α = (α₁, …, αₙ)` with `|α| ≤ order` in **graded lexicographic order**.
"Graded" means you group by total order first, then within each group you sort
lexicographically (like dictionary order, reading left to right).

**Full enumeration for `Jet<3, 2>` (3 variables, order 2):**

```
Index | Multi-index (α₁, α₂, α₃) | Total order | Monomial
------|-----------------------------|-------------|----------
  0   | (0, 0, 0)                  |     0       | 1          (constant term)
  1   | (1, 0, 0)                  |     1       | ε₁
  2   | (0, 1, 0)                  |     1       | ε₂
  3   | (0, 0, 1)                  |     1       | ε₃
  4   | (2, 0, 0)                  |     2       | ε₁²
  5   | (1, 1, 0)                  |     2       | ε₁ε₂
  6   | (1, 0, 1)                  |     2       | ε₁ε₃
  7   | (0, 2, 0)                  |     2       | ε₂²
  8   | (0, 1, 1)                  |     2       | ε₂ε₃
  9   | (0, 0, 2)                  |     2       | ε₃²
```

That's 10 entries = C(3+2, 2) = C(5, 2) = 10. ✓

This is exactly what your `multIndices` vector in the current `JetNumbers.cpp`
skeleton is intended to hold.

**Recursive algorithm to generate this table:**

The idea is to recursively fill each slot of the multi-index array. For each
variable position `pos`, try all exponents from `remaining` down to `0`, where
`remaining` is the maximum total order still available:

```cpp
// Generates all multi-indices with |α| ≤ maxOrder for `Vars` variables
// Results are stored in `table` in graded lexicographic order
void enumerate(int pos, int remaining, std::array<int, Vars>& current,
               std::vector<std::array<int, Vars>>& table)
{
    if (pos == Vars)
    {
        // We've assigned all variables — add this multi-index
        table.push_back(current);
        return;
    }
    // Try each possible exponent for variable `pos`
    for (int exp = 0; exp <= remaining; ++exp)
    {
        current[pos] = exp;
        enumerate(pos + 1, remaining - exp, current, table);
    }
}

// To get graded lex order, generate separately for each total order:
std::vector<std::array<int, Vars>> allIndices;
for (int totalOrder = 0; totalOrder <= Order; ++totalOrder)
{
    // Generate all α with |α| = exactly totalOrder
    // (modify the above to require remaining == 0 at pos == Vars)
    enumerateExact(0, totalOrder, current, allIndices);
}
```

Alternatively, generate all at once and sort by `(|α|, α₁, α₂, …)` afterward.

### 3.3 The `lookupIndex` Function

Given a multi-index `α`, you need to find its position in the table. Two
approaches:

**Approach A — Linear search (simple, fine for small N):**
```cpp
int lookupIndex(const std::array<int, Vars>& alpha)
{
    for (int i = 0; i < N; ++i)
        if (multiIndex[i] == alpha) return i;
    return -1;  // not found (|α| > Order)
}
```

**Approach B — Closed-form formula (faster, no search):**
There exists a formula to compute the position directly from the exponents,
but it's tricky. The graded lexicographic position of `(α₁, α₂, …, αₙ)` is:

```
index(α) = C(|α| + n - 1, n) - C(|α| - α₁ + n - 1, n - 1) + ...
```

This is harder to get right; the linear search is recommended for a first
implementation. You can optimize later if profiling shows it's a bottleneck.

### 3.4 The `addTable` — Pre-computed Multiplication Map

The `addTable` is an `N × N` matrix where `addTable[i][j]` stores the index `k`
such that `multiIndex[k] = multiIndex[i] + multiIndex[j]` (element-wise
addition of the exponent tuples). If the sum has `|α_i + α_j| > Order`, store
`-1` to indicate truncation.

**Why you need this:** During multiplication, you compute products of every
monomial pair. Each product `ε^α · ε^β = ε^(α+β)`. You need to know *which
coefficient slot* `α + β` maps to. Without the table, you'd call `lookupIndex`
`N²` times during every multiplication — with the table, it's a single array
access.

**Building it:**
```cpp
for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
    {
        std::array<int, Vars> sum;
        int totalOrder = 0;
        for (int v = 0; v < Vars; ++v)
        {
            sum[v] = multiIndex[i][v] + multiIndex[j][v];
            totalOrder += sum[v];
        }
        if (totalOrder > Order)
            addTable[i][j] = -1;       // truncated
        else
            addTable[i][j] = lookupIndex(sum);
    }
```

**Concrete `addTable` for `Jet<2, 2>` (6 coefficients):**

Using our index assignments: 0→(0,0), 1→(1,0), 2→(0,1), 3→(2,0), 4→(1,1), 5→(0,2)

```
addTable[i][j]:
      j=0  j=1  j=2  j=3  j=4  j=5
i=0 [  0    1    2    3    4    5  ]   (0,0)+(anything) = (anything)
i=1 [  1    3    4   -1   -1   -1 ]   (1,0)+(1,0)=(2,0)=3, (1,0)+(0,1)=(1,1)=4
i=2 [  2    4    5   -1   -1   -1 ]   (0,1)+(1,0)=(1,1)=4, (0,1)+(0,1)=(0,2)=5
i=3 [  3   -1   -1   -1   -1   -1 ]   (2,0)+(anything with |α|>0) exceeds order 2
i=4 [  4   -1   -1   -1   -1   -1 ]   (1,1) can only combine with (0,0)
i=5 [  5   -1   -1   -1   -1   -1 ]   (0,2) can only combine with (0,0)
```

Notice the pattern: row 0 is `[0,1,2,3,4,5]` because adding `(0,0)` to
anything gives the same thing back. The higher-order rows are mostly `-1`
because combining two high-order monomials exceeds the truncation limit.

### 3.5 Recommended Struct Layout

```cpp
template <std::size_t Vars, std::size_t Order>
struct Jet
{
    // Total number of coefficients = C(Vars + Order, Order)
    static constexpr std::size_t N = binomial(Vars + Order, Order);

    double coeffs[N];   // or std::array<double, N>

    // --- Lookup tables (shared across all Jet instances) ---

    // multiIndex[i] = the α-tuple for coefficient i
    static const std::array<std::array<int, Vars>, N> multiIndex;

    // addTable[i][j] = index k such that multiIndex[k] = multiIndex[i] + multiIndex[j]
    //                   or -1 if |α_i + α_j| > Order (truncated away)
    static const std::array<std::array<int, N>, N> addTable;
};
```

**Why `static`?** Every `Jet<2, 2>` instance shares the same multi-index
enumeration and the same addTable. There's no reason to duplicate this data
per-instance. Making them `static` means they're built once (at program
startup or compile time) and shared across all Jet values.

**Why `std::array` instead of `std::vector`?** Since `Vars` and `Order` are
template parameters (known at compile time), `N` is also known at compile time.
Using fixed-size arrays avoids heap allocation and enables the compiler to
optimize more aggressively. Your original skeleton used `std::vector` for both
`coeffs` and `multIndices` — switching to `std::array` is a key improvement.

---

## 4. Arithmetic Operations

### 4.1 Addition / Subtraction

Component-wise — identical to dual numbers but on all N coefficients:

```cpp
Jet operator+(const Jet& a, const Jet& b)
{
    Jet result;
    for (std::size_t i = 0; i < N; ++i)
        result.coeffs[i] = a.coeffs[i] + b.coeffs[i];
    return result;
}

Jet operator-(const Jet& a, const Jet& b)
{
    Jet result;
    for (std::size_t i = 0; i < N; ++i)
        result.coeffs[i] = a.coeffs[i] - b.coeffs[i];
    return result;
}

// Unary negation
Jet operator-(const Jet& a)
{
    Jet result;
    for (std::size_t i = 0; i < N; ++i)
        result.coeffs[i] = -a.coeffs[i];
    return result;
}
```

**Why this works:** Addition in a polynomial ring is always coefficient-wise.
If `A = Σ a_α ε^α` and `B = Σ b_α ε^α`, then `A + B = Σ (a_α + b_α) ε^α`.
This is identical to how your dual number `operator+` adds `realPart` and
`infinitesimalPart` separately — Jets just have more parts.

### 4.2 Multiplication (Cauchy Product on Multi-Indices)

This is the heart of the Jet. It generalizes the dual-number product rule.

**The math:** When you multiply two truncated polynomials:
```
A · B = (Σ a_α ε^α) · (Σ b_β ε^β) = Σ a_α · b_β · ε^(α+β)
```
For each result coefficient `c_γ`, you sum over all pairs `(α, β)` where
`α + β = γ`:
```
c_γ = Σ_{α+β=γ} a_α · b_β
```

**Compare to your dual number multiplication:**
In your `DualNumbers.cpp`, the product `(a + bε)(l + mε) = al + (am + bl)ε`
has exactly two terms in the ε-coefficient: `a·m` and `b·l`. These correspond
to the pairs `(0,1) + (1,0) → (1)` and `(1,0) + (0,1) → (1)` — but since
we only have one variable and `ε²` is truncated, there's nothing else. The Jet
multiplication is the same pattern with more multi-index pairs.

```cpp
Jet operator*(const Jet& a, const Jet& b)
{
    Jet result{};   // zero-initialize all coefficients
    for (std::size_t i = 0; i < N; ++i)
    {
        if (a.coeffs[i] == 0.0) continue;        // skip zeros for speed
        for (std::size_t j = 0; j < N; ++j)
        {
            if (b.coeffs[j] == 0.0) continue;
            int k = addTable[i][j];               // index of α_i + α_j
            if (k < 0) continue;                  // truncated
            result.coeffs[k] += a.coeffs[i] * b.coeffs[j];
        }
    }
    return result;
}
```

**Worked example — multiplying two `Jet<2, 2>` values:**

Let `A = [3, 1, 0, 0, 0, 0]` (representing `x = 3 + ε₁`) and
`B = [3, 1, 0, 0, 0, 0]` (same thing). Computing `A * B = x²`:

The loop visits all `(i, j)` pairs. Non-zero contributions:
- `i=0, j=0`: `a[0]*b[0] = 9` → `addTable[0][0] = 0` → `result[0] += 9`
- `i=0, j=1`: `a[0]*b[1] = 3` → `addTable[0][1] = 1` → `result[1] += 3`
- `i=1, j=0`: `a[1]*b[0] = 3` → `addTable[1][0] = 1` → `result[1] += 3`
- `i=1, j=1`: `a[1]*b[1] = 1` → `addTable[1][1] = 3` → `result[3] += 1`

Final result: `[9, 6, 0, 1, 0, 0]` = `9 + 6ε₁ + ε₁²`
Which is `(3 + ε₁)² = 9 + 6ε₁ + ε₁²`. Correct! ✓

> **Coefficient convention note:**
> Store `c_α = (1/α!) · ∂^α f` (Taylor coefficients). Then the Cauchy product
> loop above works with **no extra factors**, because the product of two Taylor
> series is naturally a convolution over multi-indices.

### 4.3 Division

There are two approaches. Both produce the same result.

**Approach A — Geometric series (simpler to understand):**

Split `b = b₀ + b̃` where `b₀ = b.coeffs[0]` is the real part and `b̃` is the
perturbation (all higher-order terms). Then:

```
1/b = (1/b₀) · 1/(1 + b̃/b₀)
    = (1/b₀) · Σ_{k=0}^{order} (-b̃/b₀)^k
```

This works because `b̃` has no constant term, so `b̃^k` for `k > order` is
automatically zero in the truncated algebra.

```cpp
Jet reciprocal(const Jet& b)
{
    double b0 = b.coeffs[0];
    Jet btilde = b;
    btilde.coeffs[0] = 0.0;                     // perturbation
    Jet neg_btilde_over_b0 = btilde * constant(-1.0 / b0);

    Jet result = constant(1.0);                  // k=0 term
    Jet power = constant(1.0);                   // (-b̃/b₀)^0
    for (std::size_t k = 1; k <= Order; ++k)
    {
        power = power * neg_btilde_over_b0;      // (-b̃/b₀)^k
        result = result + power;
    }
    return result * constant(1.0 / b0);
}

Jet operator/(const Jet& a, const Jet& b)
{
    return a * reciprocal(b);
}
```

**Approach B — Forward substitution (more efficient):**

Solve `b · q = a` for `q` directly. Process multi-indices in order of
increasing total degree. For each `γ`, solve:
```
q_γ = (1/b₀) · (a_γ - Σ_{α+β=γ, β≠γ} b_α · q_β)
```

### 4.4 Scalar-Jet Operations

```cpp
// Helper: create a Jet with only a constant term
static Jet constant(double c)
{
    Jet j{};
    j.coeffs[0] = c;
    return j;
}

// double + Jet: only affects the constant term
Jet operator+(double s, const Jet& a)
{
    Jet result = a;
    result.coeffs[0] += s;
    return result;
}
Jet operator+(const Jet& a, double s) { return s + a; }

// double * Jet: scales every coefficient
Jet operator*(double s, const Jet& a)
{
    Jet result;
    for (std::size_t i = 0; i < N; ++i)
        result.coeffs[i] = s * a.coeffs[i];
    return result;
}
Jet operator*(const Jet& a, double s) { return s * a; }

// double - Jet and Jet - double
Jet operator-(double s, const Jet& a) { return s + (-a); }
Jet operator-(const Jet& a, double s) { return a + (-s); }

// double / Jet and Jet / double
Jet operator/(const Jet& a, double s) { return (1.0/s) * a; }
Jet operator/(double s, const Jet& a) { return s * reciprocal(a); }
```

**Why scalar overloads matter:** Without these, expressions like `2.0 * x` or
`x + 1.0` inside your templated function `f` would fail to compile because
the compiler can't implicitly convert `double` to `Jet`. Your existing
`DualNumbers.cpp` has the implicit constructor `Dual(double, double = 0)`
which handles this, but explicit overloads are cleaner and avoid accidental
conversions.

---

## 5. Transcendental Functions

### 5.1 The General Pattern — Perturbation Decomposition

Every transcendental function on a Jet follows the same pattern:

1. **Extract** the real part `u₀ = u.coeffs[0]`.
2. **Compute** the perturbation `ũ = u - u₀` (a Jet with zero constant term).
3. **Expand** the function around `u₀` as a Taylor series in `ũ`.
4. **Truncate** — since `ũ` has no constant term, `ũ^k` is automatically zero
   for `k > Order` in the truncated algebra. So the series is finite.
5. **Return** the result, combining the real evaluation with the series.

**Why this works:** The key property of `ũ` is that `ũ.coeffs[0] = 0`. This
means every monomial in `ũ` has total order ≥ 1. Therefore `ũ²` has all
monomials of order ≥ 2, `ũ³` has order ≥ 3, etc. Once you reach `ũ^(Order+1)`,
every monomial has order > `Order` and gets truncated to zero. The series
terminates naturally.

**Compare to your dual number approach:** In `DualNumbers.cpp`, your `sin`
function does:
```cpp
Dual sin(Dual const &other) {
    return Dual(std::sin(other.realPart),
                other.infinitesimalPart * std::cos(other.realPart));
}
```
This is manually applying `d/dx sin(x) = cos(x)` and multiplying by the
chain rule factor. For Jets, you can't manually write out every derivative
order — instead, you use the Taylor series which automatically generates all
orders through the truncated polynomial arithmetic.

### 5.2 `exp` — Exponential

```
exp(u) = exp(u₀) · exp(ũ)
       = exp(u₀) · (1 + ũ + ũ²/2! + ũ³/3! + ... + ũ^Order / Order!)
```

```cpp
Jet exp(const Jet& u)
{
    double u0 = u.coeffs[0];
    Jet utilde = u;
    utilde.coeffs[0] = 0.0;           // perturbation part

    Jet result = Jet::constant(1.0);   // start with 1
    Jet power = Jet::constant(1.0);    // ũ^0
    double factorial = 1.0;

    for (std::size_t k = 1; k <= Order; ++k)
    {
        power = power * utilde;        // ũ^k  (auto-truncated by Jet multiply)
        factorial *= k;
        result = result + power * Jet::constant(1.0 / factorial);
    }

    return result * Jet::constant(std::exp(u0));
}
```

### 5.3 `log` — Natural Logarithm

```
log(u) = log(u₀) + log(1 + ũ/u₀)
       = log(u₀) + Σ_{k=1}^{Order} (-1)^(k+1) · (ũ/u₀)^k / k
```

This is the Mercator series for `log(1 + x)` with `x = ũ/u₀`.

```cpp
Jet log(const Jet& u)
{
    double u0 = u.coeffs[0];
    Jet utilde_over_u0 = u;
    utilde_over_u0.coeffs[0] = 0.0;
    utilde_over_u0 = utilde_over_u0 * Jet::constant(1.0 / u0);

    Jet result = Jet::constant(std::log(u0));
    Jet power = Jet::constant(1.0);
    for (std::size_t k = 1; k <= Order; ++k)
    {
        power = power * utilde_over_u0;   // (ũ/u₀)^k
        double sign = (k % 2 == 1) ? 1.0 : -1.0;
        result = result + power * Jet::constant(sign / k);
    }
    return result;
}
```

### 5.4 `sin` and `cos` — Trigonometric Functions

Use the angle-addition identities:
```
sin(u₀ + ũ) = sin(u₀)·cos(ũ) + cos(u₀)·sin(ũ)
cos(u₀ + ũ) = cos(u₀)·cos(ũ) - sin(u₀)·sin(ũ)
```

Where `cos(ũ)` and `sin(ũ)` are Taylor series with only even/odd terms:
```
sin(ũ) = ũ - ũ³/3! + ũ⁵/5! - ...
cos(ũ) = 1 - ũ²/2! + ũ⁴/4! - ...
```

```cpp
// Helper: compute sin(ũ) and cos(ũ) for a perturbation (ũ.coeffs[0] == 0)
std::pair<Jet, Jet> sincos_perturbation(const Jet& utilde)
{
    Jet s = Jet::constant(0.0);   // sin(ũ) starts at 0
    Jet c = Jet::constant(1.0);   // cos(ũ) starts at 1
    Jet power = Jet::constant(1.0);
    double factorial = 1.0;

    for (std::size_t k = 1; k <= Order; ++k)
    {
        power = power * utilde;
        factorial *= k;
        double coeff = 1.0 / factorial;
        switch (k % 4)
        {
            case 1: s = s + power * Jet::constant(coeff);  break; // +ũ^k/k!
            case 2: c = c - power * Jet::constant(coeff);  break; // -ũ^k/k!
            case 3: s = s - power * Jet::constant(coeff);  break; // -ũ^k/k!
            case 0: c = c + power * Jet::constant(coeff);  break; // +ũ^k/k!
        }
    }
    return {s, c};
}

Jet sin(const Jet& u)
{
    double u0 = u.coeffs[0];
    Jet utilde = u;
    utilde.coeffs[0] = 0.0;
    auto [s, c] = sincos_perturbation(utilde);
    return Jet::constant(std::sin(u0)) * c + Jet::constant(std::cos(u0)) * s;
}

Jet cos(const Jet& u)
{
    double u0 = u.coeffs[0];
    Jet utilde = u;
    utilde.coeffs[0] = 0.0;
    auto [s, c] = sincos_perturbation(utilde);
    return Jet::constant(std::cos(u0)) * c - Jet::constant(std::sin(u0)) * s;
}
```

### 5.5 `pow` — Power Function

For `pow(u, n)` where `n` is a constant (integer or real):

```
u^n = u₀^n · (1 + ũ/u₀)^n
```

Expand `(1 + x)^n` using the generalized binomial series:
```
(1 + x)^n = 1 + nx + n(n-1)x²/2! + n(n-1)(n-2)x³/3! + ...
```

```cpp
Jet pow(const Jet& u, double n)
{
    double u0 = u.coeffs[0];
    Jet x = u;
    x.coeffs[0] = 0.0;
    x = x * Jet::constant(1.0 / u0);  // x = ũ/u₀

    Jet result = Jet::constant(1.0);
    Jet power = Jet::constant(1.0);
    double binom_coeff = 1.0;           // running product: n(n-1)...(n-k+1)/k!

    for (std::size_t k = 1; k <= Order; ++k)
    {
        binom_coeff *= (n - (double)(k - 1)) / (double)k;
        power = power * x;
        result = result + power * Jet::constant(binom_coeff);
    }
    return result * Jet::constant(std::pow(u0, n));
}
```

### 5.6 `sqrt` — Square Root

`sqrt(u)` is just `pow(u, 0.5)`. You can either call `pow` directly or
specialize for efficiency:

```cpp
Jet sqrt(const Jet& u)
{
    return pow(u, 0.5);
}
```

### 5.7 `tan`, `csc`, `sec`, `cot`

These can all be built from `sin`, `cos`, and division:

```cpp
Jet tan(const Jet& u) { return sin(u) / cos(u); }
Jet csc(const Jet& u) { return Jet::constant(1.0) / sin(u); }
Jet sec(const Jet& u) { return Jet::constant(1.0) / cos(u); }
Jet cot(const Jet& u) { return cos(u) / sin(u); }
```

This is simpler than your `DualNumbers.cpp` approach where you manually derived
each trig function's derivative. With Jets, once you have `sin`, `cos`, and
`operator/`, the rest come for free.

### 5.8 Summary Table — Jet vs. Dual Function Comparison

| Function | Your Dual version (manual derivative) | Jet version (auto from series) |
|----------|--------------------------------------|-------------------------------|
| `sin(x)` | `Dual(sin(a), b*cos(a))` | Taylor series of `sin(u₀+ũ)` |
| `cos(x)` | `Dual(cos(a), -b*sin(a))` | Taylor series of `cos(u₀+ũ)` |
| `exp(x)` | (not implemented) | `exp(u₀) · Σ ũ^k/k!` |
| `log(x)` | `Dual(log(a), b/a)` | `log(u₀) + Σ (-1)^(k+1)(ũ/u₀)^k/k` |
| `pow(x,n)` | `Dual(a^n, n·a^(n-1)·b)` | Generalized binomial series |
| `tan(x)` | `Dual(tan(a), b/cos²(a))` | `sin(u) / cos(u)` — automatic |

The Jet versions are more lines of code, but they handle **all derivative
orders** simultaneously without needing to know the derivative formulas
in advance.

---

## 6. Seeding — Setting Up Variables

### 6.1 What Seeding Means

"Seeding" is how you tell the Jet system which numeric values the variables
have and which slot each variable occupies. You're creating the Jet
representation of each input variable before passing them into your function.

**Contrast with your dual number approach:** In `DualNumbers.cpp`, seeding is
implicit. You create `Dual(2)` for `x = 2` and add `Dual(0, 1)` to whichever
variable you want the derivative of. With Jets, you seed **all** variables
at once, and the derivative information propagates for **all** of them
simultaneously.

### 6.2 The `makeVariable` Function

```cpp
// Create the Jet for variable i, evaluated at value v
Jet makeVariable(std::size_t varIndex, double v)
{
    Jet j{};                           // all coefficients = 0
    j.coeffs[0] = v;                   // real part = the evaluation point
    j.coeffs[1 + varIndex] = 1.0;     // ∂xᵢ/∂xᵢ = 1  (the εᵢ coefficient)
    return j;
}
```

**Why `1 + varIndex`?** In graded lexicographic order, index 0 is always
the constant term `(0, 0, ..., 0)`. Indices 1 through `vars` are the
first-order terms `(1,0,...,0)`, `(0,1,...,0)`, etc. So variable `i`
corresponds to index `1 + i`.

**Why set it to `1.0`?** Because `∂xᵢ/∂xᵢ = 1`. The εᵢ coefficient of
variable `xᵢ` is 1 because the variable is itself with respect to its own
perturbation. This is the same reason you use `Dual(0, 1)` for ε in your
dual numbers.

### 6.3 Usage — Single Function Call for Everything

```cpp
// Your function template works with both Dual and Jet!
template <typename T>
T f(const T& x, const T& y, const T& z)
{
    return sin(x) * cos(y) * tan(z);
}

// --- With Dual Numbers (your current approach): 3 calls ---
Dual dx = f(Dual(2) + Dual(0,1), Dual(3), Dual(4));   // ∂f/∂x only
Dual dy = f(Dual(2), Dual(3) + Dual(0,1), Dual(4));   // ∂f/∂y only
Dual dz = f(Dual(2), Dual(3), Dual(4) + Dual(0,1));   // ∂f/∂z only

// --- With Jet<3, 2> (new approach): 1 call ---
using J = Jet<3, 2>;
J x = J::makeVariable(0, 2.0);   // x = 2 + ε₁
J y = J::makeVariable(1, 3.0);   // y = 3 + ε₂
J z = J::makeVariable(2, 4.0);   // z = 4 + ε₃

J result = f(x, y, z);           // ONE call: all partials through order 2
```

The `result.coeffs` array now contains **everything** — the function value,
all first partials, and all second partials (including mixed):

| Index | Multi-index α | What it stores | To get the derivative |
|-------|---------------|----------------|----------------------|
| 0     | (0,0,0)       | f(2,3,4)       | Read directly |
| 1     | (1,0,0)       | ∂f/∂x          | Read directly (α! = 1) |
| 2     | (0,1,0)       | ∂f/∂y          | Read directly (α! = 1) |
| 3     | (0,0,1)       | ∂f/∂z          | Read directly (α! = 1) |
| 4     | (2,0,0)       | (1/2!) · ∂²f/∂x² | Multiply by 2! = 2 |
| 5     | (1,1,0)       | ∂²f/∂x∂y      | Multiply by 1!·1! = 1 |
| 6     | (1,0,1)       | ∂²f/∂x∂z      | Multiply by 1!·1! = 1 |
| 7     | (0,2,0)       | (1/2!) · ∂²f/∂y² | Multiply by 2! = 2 |
| 8     | (0,1,1)       | ∂²f/∂y∂z      | Multiply by 1!·1! = 1 |
| 9     | (0,0,2)       | (1/2!) · ∂²f/∂z² | Multiply by 2! = 2 |

> **This is the entire gradient, Hessian, and beyond — from one function call.**
>
> The gradient vector is `[coeffs[1], coeffs[2], coeffs[3]]`.
> The Hessian matrix is:
> ```
> H = [ coeffs[4]*2   coeffs[5]    coeffs[6]   ]
>     [ coeffs[5]     coeffs[7]*2  coeffs[8]   ]
>     [ coeffs[6]     coeffs[8]    coeffs[9]*2 ]
> ```
> (Note: the Hessian is symmetric, so `∂²f/∂x∂y = ∂²f/∂y∂x`.)

---

## 7. Extracting Derivatives

### 7.1 The General Extraction Formula

The stored coefficient for multi-index α is `(1/α!) · ∂^α f`. To get the
actual derivative, multiply by `α!`:

```cpp
// Compute n!
constexpr double factorial(int n)
{
    double result = 1.0;
    for (int i = 2; i <= n; ++i)
        result *= i;
    return result;
}

// Get the derivative ∂^|α| f / ∂x₁^α₁ ... ∂xₙ^αₙ
double getDerivative(const Jet& jet, std::array<int, Vars> alpha)
{
    int idx = lookupIndex(alpha);       // find which coefficient slot
    if (idx < 0) return 0.0;            // order too high
    double factorialAlpha = 1.0;
    for (int a : alpha)
        factorialAlpha *= factorial(a);
    return jet.coeffs[idx] * factorialAlpha;
}
```

### 7.2 Convenience Helpers

For common use cases, add named methods:

```cpp
// Get the function value f(a₁, a₂, ...)
double value(const Jet& jet) { return jet.coeffs[0]; }

// Get first partial ∂f/∂xᵢ  (no factorial needed since 1! = 1)
double partial(const Jet& jet, int varIndex) { return jet.coeffs[1 + varIndex]; }

// Get second partial ∂²f/∂xᵢ∂xⱼ
double partial2(const Jet& jet, int var_i, int var_j)
{
    std::array<int, Vars> alpha{};
    alpha[var_i]++;
    alpha[var_j]++;
    return getDerivative(jet, alpha);   // handles the α! factor
}

// Get the gradient as a vector
std::array<double, Vars> gradient(const Jet& jet)
{
    std::array<double, Vars> grad;
    for (std::size_t i = 0; i < Vars; ++i)
        grad[i] = jet.coeffs[1 + i];
    return grad;
}

// Get the Hessian as a Vars x Vars matrix
std::array<std::array<double, Vars>, Vars> hessian(const Jet& jet)
{
    std::array<std::array<double, Vars>, Vars> H;
    for (std::size_t i = 0; i < Vars; ++i)
        for (std::size_t j = 0; j < Vars; ++j)
            H[i][j] = partial2(jet, i, j);
    return H;
}
```

### 7.3 When is the `α!` Factor Needed?

| Type of derivative | Multi-index example | α! | Factor needed? |
|-------------------|--------------------|----|---------------|
| Function value | (0, 0, 0) | 0!·0!·0! = 1 | No |
| First partial | (1, 0, 0) | 1! = 1 | No |
| Mixed second | (1, 1, 0) | 1!·1! = 1 | No |
| Pure second | (2, 0, 0) | 2! = 2 | **Yes — multiply by 2** |
| Mixed third | (1, 1, 1) | 1!·1!·1! = 1 | No |
| Pure third | (3, 0, 0) | 3! = 6 | **Yes — multiply by 6** |
| Partial (2,1,0) | (2, 1, 0) | 2!·1! = 2 | **Yes — multiply by 2** |

The pattern: `α! > 1` only when any single exponent in α is ≥ 2. For all
mixed partials where every exponent is 0 or 1, `α! = 1` and no correction
is needed.

---

## 8. Step-by-Step Implementation Plan

### Phase 1 — Foundations (utility functions)

**Step 1: `constexpr binomial(n, k)`**
```cpp
constexpr std::size_t binomial(std::size_t n, std::size_t k)
{
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    std::size_t result = 1;
    for (std::size_t i = 0; i < k; ++i)
    {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}
```
This lets you compute `N = binomial(Vars + Order, Order)` at compile time.

**Step 2: `enumerateMultiIndices()`** — see Section 3.2 for the recursive
algorithm. Generate all α with |α| ≤ Order in graded lexicographic order.

**Step 3: `buildAddTable()`** — see Section 3.4 for the nested loop that
pre-computes the multi-index sum lookup.

**Step 4: `lookupIndex(alpha)`** — see Section 3.3 for linear search and
closed-form alternatives.

### Phase 2 — Jet Struct and Basic Arithmetic

**Step 5:** Define the `Jet<Vars, Order>` struct with `coeffs` array and
static tables (Section 3.5).

**Step 6:** `operator+`, `operator-`, unary `operator-` (Section 4.1).

**Step 7:** `operator*` with the Cauchy product (Section 4.2). **Test this
thoroughly** — every other operation depends on correct multiplication.

**Step 8:** `reciprocal()` and `operator/` (Section 4.3).

**Step 9:** All scalar overloads (Section 4.4).

**Step 10:** `makeVariable()` and `Jet::constant()` (Section 6.2).

### Phase 3 — Transcendental Functions

**Step 11:** `exp`, `log` (Sections 5.2, 5.3).

**Step 12:** `sin`, `cos` with the `sincos_perturbation` helper (Section 5.4).

**Step 13:** `tan`, `csc`, `sec`, `cot` via division (Section 5.7).

**Step 14:** `pow`, `sqrt` (Sections 5.5, 5.6).

**Step 15:** `getDerivative()`, `gradient()`, `hessian()` (Section 7).

### Phase 4 — Testing and Validation

**Test 1 — Backward compatibility:** Use `Jet<1, 1>` and verify it produces
the same first derivatives as your `DualNumbers.cpp` for
`f(x, y, z) = sin(x) * cos(y) * tan(z)` at `(2, 3, 4)`.

**Test 2 — Second-order pure partials:** Compute `f(x, y) = x²y` at `(3, 2)`
with `Jet<2, 2>`. Verify:
- `∂²f/∂x² = 2y = 4`
- `∂²f/∂y² = 0`

**Test 3 — Mixed partials:** Same function. Verify `∂²f/∂x∂y = 2x = 6`.
Also verify symmetry: `∂²f/∂x∂y == ∂²f/∂y∂x`.

**Test 4 — Transcendental:** `f(x) = sin(x)` at `x = 0` with `Jet<1, 4>`.
Expected stored coefficients (Taylor coefficients of sin):
- Index 0: `sin(0) = 0`
- Index 1: `cos(0) = 1` (∂f/∂x)
- Index 2: `-sin(0)/2! = 0` (∂²f/∂x² / 2!)
- Index 3: `-cos(0)/3! = -1/6` (∂³f/∂x³ / 3!)
- Index 4: `sin(0)/4! = 0` (∂⁴f/∂x⁴ / 4!)

**Test 5 — Stress test:** Use `Jet<3, 3>` (20 coefficients) with a complex
function and verify all third-order partials against a symbolic math tool
(e.g., Wolfram Alpha or SymPy).

---

## 9. Complexity and Performance Notes

| Operation | Cost |
|-----------|------|
| Addition  | O(N) |
| Multiplication | O(N²) worst case, but many addTable entries are -1 so effectively sparser |
| Transcendental | O(Order · N²) — Order multiplications of N-element jets |

For large `vars` or `order`, N grows combinatorially. Practical sweet spots:
- 2–5 variables, order 2–4: very fast
- 10+ variables, order 3+: consider sparse representations or reverse-mode AD

---

## 10. Key Differences from Your Current Dual Numbers

| Feature | Dual Numbers | Jet Numbers |
|---------|-------------|-------------|
| Variables supported | 1 at a time | N simultaneously |
| Derivative order | 1st only | Up to `order` |
| Mixed partials | ✗ | ✓ |
| Function calls needed | N (one per variable) | **1** |
| Storage per number | 2 doubles | C(N+order, order) doubles |
| Multiplication | O(1) | O(N²) |

---

## 11. Reference: Your Existing Skeleton → What Goes Where

Your current `JetNumbers.cpp` has the right idea:

```cpp
template <std::size_t vars, std::size_t order>
struct Jet
{
    std::vector<double> coeffs;                       // → make this std::array<double, N>
    std::vector<std::array<int, vars>> multIndices;   // → make this a static lookup table
};
```

Changes to make:
- `coeffs` → `std::array<double, N>` where `N = C(vars+order, order)` (fixed size, stack-allocated).
- `multIndices` → **static** member, built once, shared by all Jet instances.
- Add the `addTable` static member for fast multiplication.
- Add constructors, arithmetic operators, and transcendental function overloads as described above.
