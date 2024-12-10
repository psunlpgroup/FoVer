# https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024

from src.downstream_evaluation.prompts.typing import FewShotPrompt


aime_fewshot_prompt: list[FewShotPrompt] = [
    # 2010-I-1
    {
        "problem": r"Maya lists all the positive divisors of $2010^2$ . She then randomly selects two distinct divisors from this list. Let $p$ be the probability that exactly one of the selected divisors is a perfect square. The probability $p$ can be expressed in the form $\frac {m}{n}$ , where $m$ and $n$ are relatively prime positive integers. Find $m + n$ .",
        "solution": r"""The prime factorization of $2010^2$ is $67^2 \cdot 3^2 \cdot 2^2 \cdot 5^2$.
Therefore, the number of divisors of $2010^2$ is $3^4$ or $81$.
A divisor of a square number is a perfect square if and only if all the exponents in its prime factorization are even.
For a divisor to be a perfect square, we can choose each exponent to be either $0$ or $2$ (both even).
Therefore, the number of perfect square divisors is $2^4 = 16$.
The number of ways we can choose 1 perfect square from the two distinct divisors is $\binom{16}{1} \binom{81-16}{1}$.
The total number of ways to pick two divisors is $\binom{81}{2}$.
Thus, the probability is: $\frac {\binom{16}{1}\binom{81-16}{1}}{\binom{81}{2}} = \frac {16\cdot65}{81\cdot40} = \frac {26}{81}$.
The final answer is $26 + 81 = 107$.""",
        "answer": "107"
    },
    # 2010-II-1
    {
        "problem": r"Let $N$ be the greatest integer multiple of $36$ all of whose digits are even and no two of whose digits are the same. Find the remainder when $N$ is divided by $1000$ .",
        "solution": r"""If an integer is divisible by $36$, it must also be divisible by $9$ since $9$ is a factor of $36$.
It is a well-known fact that, if $N$ is divisible by $9$, the sum of the digits of $N$ is a multiple of $9$.
Hence, if $N$ contains all the even digits, the sum of the digits would be $0 + 2 + 4 + 6 + 8 = 20$, which is not divisible by $9$ and thus $36$.
The next logical try would be $8640$, which happens to be divisible by $36$.
Thus, $N = 8640 \equiv 640 \pmod {1000}$.
The final answer is $640$.""",
        "answer": "640"
    },
    # 2010-I-2
    {
        "problem": r"Find the remainder when $9 \times 99 \times 999 \times \cdots \times \underbrace{99\cdots9}_{\text{999 9's}}$ is divided by $1000$ .",
        "solution": r"""Note that $999 \equiv 9999 \equiv \dots \equiv \underbrace{99\cdots9}_{\text{999 9's}} \equiv -1 \pmod{1000}$.
That is a total of $999 - 3 + 1 = 997$ integers, so all those integers multiplied out are congruent to $-1 \pmod{1000}$.
Thus, the entire expression is congruent to $-1 \times 9 \times 99 = - 891 \equiv 109 \pmod{1000}$.""",
        "answer": "109"
    },
    # 2010-II-2
    {
        "problem": r"A point $P$ is chosen at random in the interior of a unit square $S$ . Let $d(P)$ denote the distance from $P$ to the closest side of $S$ . The probability that $\frac{1}{5}\le d(P)\le\frac{1}{3}$ is equal to $\frac{m}{n}$ , where $m$ and $n$ are relatively prime positive integers. Find $m+n$ .",
        "solution": r"""$d(P) \geq \frac{1}{5}$ is a square with side length $\frac{3}{5}$ in side the unit square that has the same center, so the probability that $d(P) \geq \frac{1}{5}$ is $\left(\frac{3}{5}\right)^2=\frac{9}{25}$.
Then, $d(P) \geq \frac{1}{3}$ is a square inside $d(P) \geq \frac{1}{5}$ with side length $\frac{1}{3}$, so the probability that $d(P) \geq \frac{1}{3}$ is $\left(\frac{1}{3}\right)^2=\frac{1}{9}$.
Therefore, the probability that $\frac{1}{5} \le d(P)\le \frac{1}{3}$ is $\frac{9}{25} - \frac{1}{9} = \frac{56}{225}$.
So, the answer is $56 + 225 = 281$""",
        "answer": "281"
    },
]
