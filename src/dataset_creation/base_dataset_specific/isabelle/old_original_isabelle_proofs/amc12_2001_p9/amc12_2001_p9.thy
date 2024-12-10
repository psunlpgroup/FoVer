(*
  https://github.com/openai/miniF2F/blob/main/isabelle/valid/amc12_2001_p9.thy
*)
theory amc12_2001_p9
  imports Complex_Main
begin

theorem amc12_2001_p9:
  fixes f :: "real \<Rightarrow> real"
  assumes h0 : "\<forall> x > 0. \<forall> y > 0. f (x * y) = f (x) / y"
    and h1 : "f (500) = 3"
  shows "f (600) = 5 / 2"
proof -
  have p1 : "f (600) = f (500 * (6/5))" by auto
  have p2 : "f (500 * (6/5)) =  f (500) / (6 / 5)" using h0[rule_format, of "500" "6/5"] by auto
  have p3 : "f (500) / (6 / 5) = 5/2" using h1 by auto
  show ?thesis using p1 p2 p3 by auto (* This completes the proof *)
qed

end
