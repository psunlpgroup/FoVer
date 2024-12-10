(*
  https://github.com/openai/miniF2F/blob/main/isabelle/valid/amc12_2001_p9.thy
*)
theory amc12_2001_p9_1
  imports Complex_Main
begin

theorem example:
  fixes q :: "int \<Rightarrow> nat"
  assumes h0 : "\<forall> n \<ge> 0. q (n + 1) = 2 * q (n) + 1"
    and h1 : "q (0) = 1"
  shows "q (3) = 15"
proof -
  have p1 : "q (1) = 2 * q (0) + 1" using h0[rule_format, of 0] by auto
  have p2 : "q (1 + 1) = 2 * q (1) + 1" using h0[rule_format, of 1] by auto
  have p3 : "q (2 + 1) = 2 * q (2) + 1" using h0[rule_format, of 2] by auto
  show ?thesis using h1 p1 p2 p3 by auto
qed

end
