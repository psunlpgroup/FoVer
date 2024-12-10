theorem example:
    (* Shawna's workout goal is 30 situps. *)
    assumes "(goal::nat) = 30"
    (* On Monday, Shawna was only able to do 12 situps, so she decided that she would make up for the rest on Tuesday. *)
        and "(monday_situps::nat) = 12"
    (* However, she was only able to do 19 situps on Tuesday. *)
        and "(tuesday_situps::nat) = 19"
    (* How many situps would Shawna have to do on Wednesday to meet her minimum goal and make up for the ones she didn't do? *)
        and "(wednesday_situps::nat) = (goal - monday_situps) + (goal - tuesday_situps) + goal"
    (* Final Answer -- The answer is 1. *)
    shows "wednesday_situps = 1"
proof -
    (* Shawna's goal is 30 situps per day. *)
    (* On Monday, she did 12 situps, so she was short 30 - 12 = 18 situps. *)
    have "goal - monday_situps = 18"
        sledgehammer
    (* On Tuesday, she made up for 19 situps. *)
    (* So now she is 18 - 19 = -1 short. *)
    then have "goal - monday_situps - tuesday_situps = 1"
        sledgehammer
    (* To meet her goal and make up for the ones she didn't do, she would have to do 1 more situp on Wednesday. *)
    (* Therefore, the answer (arabic numerals) is 1. *)
    then have "wednesday_situps = 1"
        sledgehammer
    thus ?thesis
        sledgehammer
qed
