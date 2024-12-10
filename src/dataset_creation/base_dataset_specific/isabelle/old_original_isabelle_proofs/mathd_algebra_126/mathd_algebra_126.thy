(*
  https://github.com/openai/miniF2F/blob/main/isabelle/valid/mathd_algebra_126.thy
*)
theory mathd_algebra_126
  imports Complex_Main
begin

theorem mathd_algebra_126:
  fixes x y :: real
  assumes h0 : "2 * 3 = x - 9"
    and h1 : "2 * (-5) = y + 1"
  shows "x = 15 \<and> y = -11"
proof -
  have p0 : "x = 15" using h0 by auto
  have p1 : "y = -11" using h1 by auto
  show ?thesis using p0 p1 by auto (* This completes the proof *)
qed

end
