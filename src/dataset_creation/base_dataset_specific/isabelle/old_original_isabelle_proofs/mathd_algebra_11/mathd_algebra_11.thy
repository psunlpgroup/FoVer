(*
  https://github.com/openai/miniF2F/blob/main/isabelle/valid/mathd_algebra_11.thy
*)
theory mathd_algebra_11 imports
  Complex_Main
begin

theorem mathd_algebra_11:
  fixes a b :: real
  assumes h0 : "a \<noteq> b"
    and h1 : "a \<noteq> 2 * b"
    and h2 : "(4*a + 3*b) / (a - 2*b) = 5"
  shows "(a + 11*b) / (a-b) = 2"
proof -
  have p0 : "a-b \<noteq> 0" using h0 by auto
  have "a - 2*b \<noteq> 0" using h1 by auto
  hence "(4*a+3*b) = 5 * (a-2*b)" using h2 by (metis Groups.mult_ac(2) nonzero_mult_div_cancel_left times_divide_eq_right)
  also have "5 * (a - 2*b) = 5 * a - 10 * b" by auto
  ultimately have "4*a + 3*b = 5 * a - 10 * b" by auto
  hence p1 : "a = 13*b" by auto
  have "a + 11 * b = 2 * (a-b)" using p0 unfolding p1 by auto
  then show ?thesis using p0 by auto (* This completes the proof *)
qed

end
