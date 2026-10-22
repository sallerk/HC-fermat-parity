# HC-fermat-parity

Code and data supporting the note *Least lengths of Aoki's exceptional Hodge classes on Fermat varieties, and
degrees with four prime factors* (K. Saller, 2026), DOI: [10.5281/zenodo.23571028](https://doi.org/10.5281/zenodo.23571028).

**Subject.** Aoki reduced the Hodge conjecture for Fermat varieties to the algebraicity of the classes of
exceptional Hodge elements xi_q, and Miranda, Movasati, Rufino and Villaflor deduced from a result of Kang that
such a class is algebraic once it is congruent, modulo linear and standard elements, to an element with at most
six entries. For q = lp and q = 4p the note describes the Hodge elements by pairs of odd functions, determines the least length of the class
of xi_q in closed form, and bounds the lengths of its lifts to the degrees 2q and rq. For q with four prime
factors, such as 420, it bounds the least length of a Hodge element on which a parity invariant (in other terms
due to Kubert) is nonzero, determines it as 15 at 420 and 16 at 924, and shows that Aoki's recursion for xi_q has
no solution.

**Role of the computations.** The note's section "Computations" lists eight items. Most of them check statements
that the note proves by hand; some results rest on computations, and the note says so where they are stated:
values of lifted lengths (integer programs as upper bounds, an exhaustive search for lower bounds), the classes
at 420, and a search over the degrees below 20000 with four prime factors.
[`README.txt`](README.txt) gives the requirements, the commands, what each script checks, how the scripts' labels
translate into the numbering of the note, and the result of the rerun made when this repository was assembled.

**Status:** not yet peer reviewed.

**Disclosure:** the arguments, the computations and the code were produced with the assistance of an AI system
(Claude, by Anthropic). The scripts in `fermat-parity/checks` and `fermat-parity/certify` were written later,
partly from the text of the note or from Aoki's article without the other code, and repeat some of its checks;
they were also produced with AI assistance, so they are not independent in the usual sense.

## Licences

- **Code** (`*.py`, `*.sh`): MIT License, see [`LICENSE`](LICENSE).
- **Everything else** (README files and computed outputs): Creative Commons Attribution 4.0 International
  (CC BY 4.0), see [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0).
- No third-party files are included.
