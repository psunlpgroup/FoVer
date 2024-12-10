theory Interactive imports
  HOL.HOL Complex_Main 
  "HOL-Library.Code_Target_Numeral" 
  "HOL-Library.Sum_of_Squares" 
  "Symmetric_Polynomials.Vieta" 
  "HOL-Computational_Algebra.Computational_Algebra" 
  "HOL-Number_Theory.Number_Theory"
begin

function digits_in_base :: "nat \<Rightarrow> nat \<Rightarrow> nat list" where 
  "digits_in_base n k = (if n div k = 0 \<or> k=1
      then [n] else (n mod k) # (digits_in_base (n div k) k))"
  by auto
termination 
  by (relation "measure fst") (auto simp add: div_greater_zero_iff)

end