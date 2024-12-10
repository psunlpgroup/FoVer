theorem example:
    (* Nancy and Jason are learning to dance for the first time. Nancy steps on her partner's feet 3 times as often as Jason. *)
    assumes "(Nancy_steps::nat) = 3 * (Jason_steps::nat)"
    (* If together they step on each other's feet 32 times, how many times does Jason step on his partner's feet? *)
        and "(Nancy_steps::nat) + (Jason_steps::nat) = 32"
    (* Final Answer -- The answer is 8. *)
    shows "(Jason_steps::nat) = 8"
proof -
    (* Let's say Jason steps on his partner's feet x times. *)
    (* Nancy steps on her partner's feet 3 times as often, so Nancy steps on her partner's feet 3x times. *)
    (* Together they step on each other's feet 32 times, so we can make the equation: *)
    (* x + 3x = 32 *)
    have "3 * Jason_steps + Jason_steps = 32" using assms
        sledgehammer
    (* Combine like terms: 4x = 32 *)
    then have "4 * Jason_steps = 32"
        sledgehammer
    (* Divide by 4: x = 8 *)
    then have "Jason_steps = 32 div 4"
        sledgehammer
    (* So Jason steps on his partner's feet 8 times. *)
    (* Therefore, the answer (arabic numerals) is 8. *)
    then have "Jason_steps = 8"
        sledgehammer
    thus ?thesis
        sledgehammer
qed
