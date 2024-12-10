(* 
  https://github.com/openai/miniF2F/blob/main/isabelle/valid/amc12a_2013_p7.thy
*)
theory amc12a_2013_p7
  imports Complex_Main
begin

theorem amc12a_2013_p7:
  fixes s :: "nat \<Rightarrow> real"
  assumes h0 : "\<And>n. s (n+2) = s (n+1) + s n"
    and h1 : "s (9) = 110"
    and h2 : "s (7) = 42"
  shows "s (4) = 10"
proof -
  have p1 : "s (9) = s (8) + s (7)" using h0[of 7] by auto
  have p2 : "s (8) = s (9) - s (7)" using p1 by auto
  have p3 : "s (8) = 110 - 42" using p2 h1 h2 by auto
  have p4 : "s (8) = 68" using p3 by auto
  have p5 : "s (8) = s (7) + s (6)" using h0[of 6] by auto
  have p6 : "s (6) = s (8) - s(7)" using p5 by auto
  have p7 : "s (6) = 68 - 42" using p4 p6 h2 by auto
  have p8 : "s (6) = 26" using p7 by auto
  have p9 : "s (7) = s (6) + s (5)" using h0[of 5] by auto
  have p10 : "s (5) = s (7) - s (6)" using p9 by auto
  have p11 : "s (5) = 42 - 26" using p10 h2 p8 by auto
  have p12 : "s (5) = 16" using p11 by auto
  have p13 : "s (6) = s (5) + s (4)" using h0[of 4] by auto
  have p14 : "s (4) = s (6) - s (5)" using p13 by auto
  have p15 : "s (4) = 26 - 16" using p14 p8 p12 by auto
  have p16 : "s (4) = 10" using p15 by auto
  show ?thesis using p16 by auto (* This completes the proof *)
qed

end
