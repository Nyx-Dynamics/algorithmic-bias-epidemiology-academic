"""
Primitive Recursive Functions: Theory and Application to Document Analysis

This module demonstrates primitive recursive function theory using the
OrganizedDocuments collection as a concrete data source.

PRIMITIVE RECURSIVE FUNCTIONS are built from three basic functions and
two operations:

BASIC FUNCTIONS:
    1. Zero:        Z() = 0
    2. Successor:   S(n) = n + 1
    3. Projection:  P_i^n(x_1, ..., x_n) = x_i

OPERATIONS:
    4. Composition: h(x) = f(g_1(x), ..., g_k(x))
    5. Primitive Recursion:
        h(0, y) = f(y)
        h(S(n), y) = g(h(n, y), n, y)

All computable functions that always terminate can be built this way.
Functions like Ackermann's function are TOTAL but NOT primitive recursive.

Author: Generated from OrganizedDocuments analysis
Data Source: /Users/acdmbpmax/OrganizedDocuments/scan_results.json
"""

import json
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Any
from functools import reduce
import sys

sys.setrecursionlimit(100)  # Keep low to demonstrate stack limits


# =============================================================================
# PART I: PRIMITIVE RECURSIVE BUILDING BLOCKS (Pure Theory)
# =============================================================================

class PrimitiveRecursiveTheory:
    """
    PURE implementation of primitive recursive functions.

    These implementations follow the exact recursive definitions
    from computability theory. They work for small inputs but will
    hit recursion limits for larger values.

    For practical computation, see PrimitiveRecursivePractical below.
    """

    # -------------------------------------------------------------------------
    # BASIC FUNCTIONS (Axioms)
    # -------------------------------------------------------------------------

    @staticmethod
    def zero() -> int:
        """Z() = 0 - The zero function."""
        return 0

    @staticmethod
    def successor(n: int) -> int:
        """S(n) = n + 1 - The successor function."""
        return n + 1

    @staticmethod
    def projection(i: int, *args) -> int:
        """P_i^n(x_1, ..., x_n) = x_i - Projection function (0-indexed)."""
        return args[i]

    # -------------------------------------------------------------------------
    # DERIVED ARITHMETIC (Pure Recursive - Small Inputs Only)
    # -------------------------------------------------------------------------

    @staticmethod
    def add_pure(a: int, b: int) -> int:
        """
        add(0, b) = b
        add(S(a), b) = S(add(a, b))
        """
        if a == 0:
            return b
        return PrimitiveRecursiveTheory.successor(
            PrimitiveRecursiveTheory.add_pure(a - 1, b)
        )

    @staticmethod
    def multiply_pure(a: int, b: int) -> int:
        """
        mult(0, b) = 0
        mult(S(a), b) = add(mult(a, b), b)
        """
        if a == 0:
            return 0
        return PrimitiveRecursiveTheory.add_pure(
            PrimitiveRecursiveTheory.multiply_pure(a - 1, b), b
        )

    @staticmethod
    def predecessor(n: int) -> int:
        """pred(0) = 0, pred(S(n)) = n"""
        return 0 if n == 0 else n - 1

    @staticmethod
    def monus_pure(a: int, b: int) -> int:
        """
        monus(a, 0) = a
        monus(a, S(b)) = pred(monus(a, b))
        """
        if b == 0:
            return a
        return PrimitiveRecursiveTheory.predecessor(
            PrimitiveRecursiveTheory.monus_pure(a, b - 1)
        )

    @staticmethod
    def factorial_pure(n: int) -> int:
        """
        fact(0) = 1
        fact(S(n)) = mult(S(n), fact(n))
        """
        if n == 0:
            return 1
        return PrimitiveRecursiveTheory.multiply_pure(
            n, PrimitiveRecursiveTheory.factorial_pure(n - 1)
        )


# =============================================================================
# PART II: PRACTICAL PRIMITIVE RECURSIVE (Native Arithmetic)
# =============================================================================

class PrimitiveRecursivePractical:
    """
    Practical implementation using native Python arithmetic.

    These are PROVABLY EQUIVALENT to the pure recursive definitions
    but use iteration/native ops for efficiency. Every function here
    is still primitive recursive in the theoretical sense.
    """

    @staticmethod
    def add(a: int, b: int) -> int:
        """Addition - primitive recursive via repeated successor."""
        return a + b  # Equivalent to a applications of successor to b

    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiplication - primitive recursive via repeated addition."""
        return a * b

    @staticmethod
    def predecessor(n: int) -> int:
        """Predecessor with 0 ↦ 0."""
        return max(0, n - 1)

    @staticmethod
    def monus(a: int, b: int) -> int:
        """Truncated subtraction: a ∸ b = max(a - b, 0)."""
        return max(0, a - b)

    @staticmethod
    def power(base: int, exp: int) -> int:
        """Exponentiation - primitive recursive via repeated multiplication."""
        return base ** exp

    @staticmethod
    def factorial(n: int) -> int:
        """Factorial - primitive recursive."""
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    @staticmethod
    def is_zero(n: int) -> int:
        """is_zero(n) = 1 if n = 0, else 0."""
        return 1 if n == 0 else 0

    @staticmethod
    def sgn(n: int) -> int:
        """Sign: sgn(0) = 0, sgn(n) = 1 for n > 0."""
        return 0 if n == 0 else 1

    @staticmethod
    def leq(a: int, b: int) -> int:
        """Less or equal: a ≤ b."""
        return 1 if a <= b else 0

    @staticmethod
    def lt(a: int, b: int) -> int:
        """Less than: a < b."""
        return 1 if a < b else 0

    @staticmethod
    def eq(a: int, b: int) -> int:
        """Equality: a = b."""
        return 1 if a == b else 0

    @staticmethod
    def AND(a: int, b: int) -> int:
        """Logical AND."""
        return min(a, b)

    @staticmethod
    def OR(a: int, b: int) -> int:
        """Logical OR."""
        return PrimitiveRecursivePractical.sgn(a + b)

    @staticmethod
    def NOT(a: int) -> int:
        """Logical NOT."""
        return 1 - min(a, 1)

    @staticmethod
    def if_then_else(cond: int, then_val: int, else_val: int) -> int:
        """Conditional: cond ? then_val : else_val."""
        return cond * then_val + (1 - cond) * else_val


# =============================================================================
# PART III: BOUNDED OPERATIONS
# =============================================================================

class BoundedOperations:
    """
    Bounded quantification and search - primitive recursive because bounded.
    """

    @staticmethod
    def bounded_sum(f: Callable[[int], int], bound: int) -> int:
        """Σ_{i=0}^{bound-1} f(i)"""
        return sum(f(i) for i in range(bound))

    @staticmethod
    def bounded_product(f: Callable[[int], int], bound: int) -> int:
        """Π_{i=0}^{bound-1} f(i)"""
        result = 1
        for i in range(bound):
            result *= f(i)
        return result

    @staticmethod
    def bounded_exists(predicate: Callable[[int], int], bound: int) -> int:
        """∃i < bound: predicate(i)."""
        return 1 if any(predicate(i) for i in range(bound)) else 0

    @staticmethod
    def bounded_forall(predicate: Callable[[int], int], bound: int) -> int:
        """∀i < bound: predicate(i)."""
        return 1 if all(predicate(i) for i in range(bound)) else 0

    @staticmethod
    def bounded_min(predicate: Callable[[int], int], bound: int) -> int:
        """μi < bound: predicate(i) - smallest i where predicate holds."""
        for i in range(bound):
            if predicate(i):
                return i
        return bound

    @staticmethod
    def bounded_count(predicate: Callable[[int], int], bound: int) -> int:
        """Count i < bound where predicate(i) holds."""
        return sum(1 for i in range(bound) if predicate(i))


# =============================================================================
# PART IV: GÖDEL ENCODING (Key Primitive Recursive Technique)
# =============================================================================

class GodelEncoding:
    """
    Gödel encoding: represent sequences as single natural numbers.

    This technique was crucial for Gödel's incompleteness theorems.
    All operations here are primitive recursive.
    """

    @staticmethod
    def prime(n: int) -> int:
        """Return the n-th prime (0-indexed). Primitive recursive."""
        primes = [2]
        candidate = 3
        while len(primes) <= n:
            is_prime = all(candidate % p != 0 for p in primes)
            if is_prime:
                primes.append(candidate)
            candidate += 2
        return primes[n]

    @staticmethod
    def encode_sequence(seq: List[int]) -> int:
        """
        Gödel encoding: ⟨a_0, a_1, ..., a_n⟩ = p_0^(a_0+1) * p_1^(a_1+1) * ...

        Each element is encoded as an exponent of a prime.
        The +1 allows encoding of 0.
        """
        if not seq:
            return 1  # Empty sequence
        result = 1
        for i, val in enumerate(seq):
            result *= GodelEncoding.prime(i) ** (val + 1)
        return result

    @staticmethod
    def decode_length(code: int) -> int:
        """
        Find the length of an encoded sequence.
        Length is highest i where p_i divides code.
        """
        if code <= 1:
            return 0
        length = 0
        i = 0
        while True:
            p = GodelEncoding.prime(i)
            if code % p != 0:
                break
            length = i + 1
            i += 1
            if p > code:
                break
        return length

    @staticmethod
    def decode_element(code: int, index: int) -> int:
        """
        Extract element at index from encoded sequence.
        Find exponent of p_index in code, subtract 1.
        """
        p = GodelEncoding.prime(index)
        exp = 0
        while code % p == 0:
            exp += 1
            code //= p
        return max(0, exp - 1)

    @staticmethod
    def beta_function(c: int, d: int, i: int) -> int:
        """
        Gödel's β function: β(c, d, i) = c mod (1 + (i+1) * d)

        This function allows finite sequences to be encoded
        using just two numbers (c, d) instead of prime factorization.

        Key property: For any sequence a_0, ..., a_n, there exist c, d
        such that β(c, d, i) = a_i for all i ≤ n.
        """
        return c % (1 + (i + 1) * d)


# =============================================================================
# PART V: APPLICATION TO DOCUMENT DATA
# =============================================================================

class DocumentAnalysis:
    """
    Primitive recursive analysis of the OrganizedDocuments collection.
    """

    def __init__(self, json_path: str = "/Users/acdmbpmax/OrganizedDocuments/scan_results.json"):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.categories = self.data['categories']
        self.total_files = self.data['total_files']
        self.total_matches = self.data['total_matches']

        self.category_list = list(self.categories.keys())
        self.count_list = list(self.categories.values())

    def count_at(self, i: int) -> int:
        """Get count at index i."""
        return self.count_list[i] if i < len(self.count_list) else 0

    # -------------------------------------------------------------------------
    # PRIMITIVE RECURSIVE OPERATIONS ON DOCUMENT DATA
    # -------------------------------------------------------------------------

    def total_categorized(self) -> int:
        """Σ counts - bounded sum."""
        return BoundedOperations.bounded_sum(self.count_at, len(self.count_list))

    def categories_above_threshold(self, threshold: int) -> int:
        """Count categories exceeding threshold."""
        predicate = lambda i: PrimitiveRecursivePractical.lt(threshold, self.count_at(i))
        return BoundedOperations.bounded_count(predicate, len(self.count_list))

    def max_category_index(self) -> int:
        """Index of largest category (bounded search)."""
        max_idx, max_val = 0, 0
        for i in range(len(self.count_list)):
            if self.count_at(i) > max_val:
                max_idx, max_val = i, self.count_at(i)
        return max_idx

    def integer_average(self) -> int:
        """Integer division: total / count."""
        total = self.total_categorized()
        count = len(self.count_list)
        return total // count if count > 0 else 0

    def gcd(self, a: int, b: int) -> int:
        """GCD via Euclidean algorithm (primitive recursive)."""
        while b:
            a, b = b, a % b
        return a

    def encode_category_counts(self) -> int:
        """Gödel-encode the category counts as a single number."""
        return GodelEncoding.encode_sequence(self.count_list)

    def decode_category_count(self, code: int, index: int) -> int:
        """Decode a category count from Gödel encoding."""
        return GodelEncoding.decode_element(code, index)


# =============================================================================
# PART VI: DEMONSTRATION
# =============================================================================

def demonstrate():
    """Full demonstration of primitive recursive functions."""

    PR = PrimitiveRecursivePractical
    PRT = PrimitiveRecursiveTheory
    BO = BoundedOperations
    GE = GodelEncoding

    print("=" * 80)
    print("PRIMITIVE RECURSIVE FUNCTION THEORY")
    print("Applied to OrganizedDocuments Collection")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: Basic Functions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 1: BASIC PRIMITIVE RECURSIVE FUNCTIONS")
    print("-" * 60)

    print(f"\nBasic Functions (Axioms):")
    print(f"  Zero: Z() = {PRT.zero()}")
    print(f"  Successor: S(5) = {PRT.successor(5)}")
    print(f"  Projection: P_1(10, 20, 30) = {PRT.projection(1, 10, 20, 30)}")

    print(f"\nDerived Arithmetic (Pure Recursive - small inputs):")
    print(f"  add(3, 4) = {PRT.add_pure(3, 4)}")
    print(f"  multiply(3, 4) = {PRT.multiply_pure(3, 4)}")
    print(f"  monus(7, 3) = {PRT.monus_pure(7, 3)}")
    print(f"  monus(3, 7) = {PRT.monus_pure(3, 7)} (bottoms at 0)")
    print(f"  factorial(3) = {PRT.factorial_pure(3)} (using small n to avoid stack overflow)")

    # -------------------------------------------------------------------------
    # PART 2: Logic
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 2: PRIMITIVE RECURSIVE LOGIC")
    print("-" * 60)

    print(f"\nPredicates (encoded as 0/1):")
    print(f"  is_zero(0) = {PR.is_zero(0)}")
    print(f"  is_zero(5) = {PR.is_zero(5)}")
    print(f"  leq(5, 7) = {PR.leq(5, 7)}")
    print(f"  leq(7, 5) = {PR.leq(7, 5)}")
    print(f"  eq(5, 5) = {PR.eq(5, 5)}")
    print(f"  lt(3, 5) = {PR.lt(3, 5)}")

    print(f"\nLogical Operations:")
    print(f"  AND(1, 1) = {PR.AND(1, 1)}")
    print(f"  AND(1, 0) = {PR.AND(1, 0)}")
    print(f"  OR(0, 1) = {PR.OR(0, 1)}")
    print(f"  NOT(1) = {PR.NOT(1)}")
    print(f"  if_then_else(1, 100, 200) = {PR.if_then_else(1, 100, 200)}")
    print(f"  if_then_else(0, 100, 200) = {PR.if_then_else(0, 100, 200)}")

    # -------------------------------------------------------------------------
    # PART 3: Bounded Operations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 3: BOUNDED OPERATIONS")
    print("-" * 60)

    print(f"\nBounded Quantifiers:")
    print(f"  Σ i for i∈[0,5): {BO.bounded_sum(lambda i: i, 5)}")
    print(f"  Π (i+1) for i∈[0,5): {BO.bounded_product(lambda i: i+1, 5)} (= 5!)")
    print(f"  ∃i<10: i=7 → {BO.bounded_exists(lambda i: PR.eq(i, 7), 10)}")
    print(f"  ∀i<5: i<10 → {BO.bounded_forall(lambda i: PR.lt(i, 10), 5)}")
    print(f"  μi<10: i>3 → {BO.bounded_min(lambda i: PR.lt(3, i), 10)}")
    print(f"  #{'{'}i<10: i is even{'}'} → {BO.bounded_count(lambda i: PR.eq(i%2, 0), 10)}")

    # -------------------------------------------------------------------------
    # PART 4: Gödel Encoding
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 4: GÖDEL ENCODING")
    print("-" * 60)

    seq = [3, 1, 4, 1, 5]
    encoded = GE.encode_sequence(seq)
    print(f"\nEncoding sequence {seq}:")
    print(f"  Gödel number: {encoded}")
    print(f"  Decoded elements:")
    for i in range(len(seq)):
        print(f"    [{i}] = {GE.decode_element(encoded, i)}")

    print(f"\nβ function (alternative encoding):")
    print(f"  β(14, 3, 0) = {GE.beta_function(14, 3, 0)}")
    print(f"  β(14, 3, 1) = {GE.beta_function(14, 3, 1)}")
    print(f"  β(14, 3, 2) = {GE.beta_function(14, 3, 2)}")

    print(f"\nFirst 10 primes: {[GE.prime(i) for i in range(10)]}")

    # -------------------------------------------------------------------------
    # PART 5: Document Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 5: DOCUMENT DATA ANALYSIS")
    print("-" * 60)

    try:
        docs = DocumentAnalysis()

        print(f"\nDocument Categories:")
        for i, (cat, count) in enumerate(docs.categories.items()):
            print(f"  [{i}] {cat}: {count}")

        print(f"\nPrimitive Recursive Statistics:")
        print(f"  Total files scanned: {docs.total_files}")
        print(f"  Total categorized (data): {docs.total_matches}")
        print(f"  Total categorized (Σ counts): {docs.total_categorized()}")
        print(f"  Categories with >100 docs: {docs.categories_above_threshold(100)}")
        print(f"  Categories with >50 docs: {docs.categories_above_threshold(50)}")

        max_idx = docs.max_category_index()
        print(f"  Largest category: [{max_idx}] {docs.category_list[max_idx]} "
              f"({docs.count_list[max_idx]} docs)")
        print(f"  Average category size: {docs.integer_average()}")

        gilead = docs.categories.get('gilead_employment', 0)
        tax = docs.categories.get('tax_returns', 0)
        g = docs.gcd(gilead, tax)
        print(f"\n  Gilead/Tax ratio: {gilead}:{tax} = {gilead//g}:{tax//g}")

        print(f"\nGödel Encoding of Category Counts:")
        code = docs.encode_category_counts()
        print(f"  Gödel number: {code}")
        print(f"  (This single integer encodes all {len(docs.count_list)} category counts)")
        print(f"  Verification - decode first 3:")
        for i in range(min(3, len(docs.count_list))):
            decoded = docs.decode_category_count(code, i)
            actual = docs.count_list[i]
            print(f"    [{i}] {docs.category_list[i]}: decoded={decoded}, actual={actual}")

    except FileNotFoundError:
        print("  (Document data not found)")

    # -------------------------------------------------------------------------
    # PART 6: Theory Summary
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 6: THEORETICAL SIGNIFICANCE")
    print("-" * 60)

    print("""
    HIERARCHY OF FUNCTIONS:

    Primitive Recursive ⊊ Total Recursive ⊊ Partial Recursive
           ↓                    ↓                  ↓
    Always terminates     Ackermann, etc.    May not terminate
    Bounded loops         Unbounded search    Halting problem

    KEY RESULTS:

    1. CLOSURE: Primitive recursive functions are closed under:
       - Composition
       - Primitive recursion
       - Bounded quantification (∃x<n, ∀x<n)

    2. EXPRESSIVENESS: Includes all "practical" arithmetic:
       +, ×, ^, !, mod, gcd, primality, divisibility...

    3. LIMITATIONS: Cannot express:
       - Ackermann function (total but grows too fast)
       - Busy beaver (not computable)
       - Halting problem (not computable)

    4. GÖDEL'S USE: Primitive recursive functions suffice to:
       - Encode/decode formulas as numbers
       - Define "provability" arithmetically
       - Prove the incompleteness theorems

    CONNECTION TO YOUR DOCUMENTS:

    The operations performed on your document collection
    (counting, summing, searching, encoding) are all
    primitive recursive - they always terminate and
    can be computed with bounded resources.

    Even the Gödel encoding of your 9 category counts
    into a single integer is primitive recursive.
    """)

    # -------------------------------------------------------------------------
    # PART 7: Ackermann (Beyond Primitive Recursive)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 7: ACKERMANN FUNCTION (Beyond Primitive Recursive)")
    print("-" * 60)

    def ackermann(m: int, n: int) -> int:
        """Iterative Ackermann using explicit stack."""
        stack = [(m, n, False, 0)]  # (m, n, waiting_for_inner, inner_result)
        result = 0

        while stack:
            m, n, waiting, inner = stack.pop()

            if waiting:
                # We have inner result, now compute A(m-1, inner)
                stack.append((m - 1, inner, False, 0))
            elif m == 0:
                result = n + 1
                # Pass result up if there's a waiting frame
                if stack and stack[-1][2]:
                    old = stack.pop()
                    stack.append((old[0], old[1], True, result))
            elif n == 0:
                stack.append((m - 1, 1, False, 0))
            else:
                # Need A(m, n-1) first, then A(m-1, result)
                stack.append((m, n, True, 0))  # Wait for inner result
                stack.append((m, n - 1, False, 0))  # Compute inner first

            if len(stack) > 10000:
                return -1  # Too deep

        return result

    # Pre-computed Ackermann values (because recursion is too deep)
    ackermann_table = {
        (0, 0): 1, (0, 1): 2, (0, 2): 3, (0, 3): 4, (0, 4): 5, (0, 5): 6,
        (1, 0): 2, (1, 1): 3, (1, 2): 4, (1, 3): 5, (1, 4): 6, (1, 5): 7,
        (2, 0): 3, (2, 1): 5, (2, 2): 7, (2, 3): 9, (2, 4): 11, (2, 5): 13,
        (3, 0): 5, (3, 1): 13, (3, 2): 29, (3, 3): 61, (3, 4): 125, (3, 5): 253,
        (4, 0): 13, (4, 1): 65533, (4, 2): "2^65536-3",
    }

    print("\nAckermann function A(m, n):")
    print("       n=0    n=1    n=2    n=3    n=4    n=5")
    for m in range(5):
        row = f"m={m}:"
        for n in range(6):
            val = ackermann_table.get((m, n), ">HUGE")
            if isinstance(val, int):
                row += f"{val:>7}"
            else:
                row += f"{val:>7}"
        print(row)

    print("""
    Ackermann is TOTAL (always terminates) but NOT primitive recursive.

    A(4,2) = 2^65536 - 3  ≈ 10^19728 digits!
    A(5,0) = 65533
    A(5,1) = already incomprehensibly large

    This proves primitive recursive ⊊ total recursive.
    """)

    print("=" * 80)
    print("END OF DEMONSTRATION")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate()
