class ScoreManager:
    """
    Tool to compute the final score in a drone rescue simulation.

     It takes into account the number of drones, time
     limits, and the number of bombs to calculate the score based on
     rescue percentage, exploration score, health returned and time taken to
     rescue all the bombs.

    Attributes:
        number_drones (int): Number of drones in the map.
        max_timestep_limit (int): Max number of timesteps.
        max_walltime_limit (int): Max wall time in seconds.
        total_number_bombs (int): Number of bombs to dispose.
        w_disposal (float): Weight for rescue score.
        w_exploration (float): Weight for exploration score.
        w_score_health_returned (float): Weight for health return score.
        w_time (float): Weight for time score.
        time_best_ratio (float): Fraction of ``max_timestep_limit`` at (or
            before) which the time score is full (default 0.5).
    """

    def __init__(self,
                 number_drones: int,
                 max_timestep_limit: int,
                 max_walltime_limit: int,
                 total_number_bombs: int,
                 exploration_threshold_for_time_bonus: float = 97.0,
                 time_best_ratio: float = 0.5):
        """
        Initialize the ScoreManager.

        Args:
            number_drones (int): Number of drones.
            max_timestep_limit (int): Max timesteps.
            max_walltime_limit (int): Max wall time in seconds.
            total_number_bombs (int): Number of bombs.
        """

        # 'number_drones' is the number of drones that will be generated in the
        # map
        self.number_drones = number_drones

        # 'max_timestep_limit' is the number of timesteps after which the
        # session will end.
        self.max_timestep_limit = max_timestep_limit

        # 'max_walltime_limit' is the elapsed time (in seconds) after which the
        # session will end.
        self.max_walltime_limit = max_walltime_limit

        # 'time_best_ratio' is the fraction of 'max_timestep_limit' at (or
        # before) which the time score is full (same convention as the red
        # team's place_scoring.time_best_ratio).
        self.time_best_ratio = float(time_best_ratio)

        # 'number_bombs' is the number of bombs that should
        # be retrieved by the drones.
        self.total_number_bombs = total_number_bombs

        # weight for the different parts of the score. The sum must be equal
        # to 1.
        self.w_disposal = 0.6
        self.w_exploration = 0.0
        self.w_score_health_returned = 0.2
        self.w_time = 0.2
        self.exploration_threshold_for_time_bonus = float(
            exploration_threshold_for_time_bonus
        )

    def compute_score(self, number_disposed_bombs: int, score_exploration: float,
                      score_health_returned: float, elapsed_timestep: int,
                      has_crashed: bool = False) -> tuple:
        """
        Compute the final score out of 100.

        The time score uses the same convention as the red team
        (``PlaceScoreManager.compute_time_score``): at or before half the
        timestep limit it is full (out of 100), it drops to 0 at the limit and
        is linear in between; a crashed round earns 0 on time. The rescue round
        ends as soon as all bombs are disposed, so ``elapsed_timestep`` equals
        the disposal-completion timestep for completing runs and reaches the
        limit for incomplete ones.

        Args:
            number_disposed_bombs (int): Number of disposed bombs.
            score_exploration (float): Exploration score.
            score_health_returned (float): Health return score.
            elapsed_timestep (int): Timesteps elapsed at the end of the round.
            has_crashed (bool): Whether the round crashed (zeroes the time
                score, like the red team).

        Returns:
            tuple: (score, percentage_disposed, score_timestep)
        """
        if self.total_number_bombs > 0:
            percentage_disposed = (number_disposed_bombs /
                                 self.total_number_bombs * 100.0)
        else:
            percentage_disposed = 100.0

        score_timestep = self._compute_time_score(elapsed_timestep, has_crashed)

        score = self.w_disposal * percentage_disposed + \
                self.w_exploration * score_exploration + \
                self.w_score_health_returned * score_health_returned + \
                self.w_time * score_timestep

        return score, percentage_disposed, score_timestep

    def _compute_time_score(self, elapsed_timestep: int,
                            has_crashed: bool = False) -> float:
        """
        Time score out of 100, mirroring the red team's time score.

        ``score_timestep = clamp((limit - elapsed) / (limit - best), 0, 1) * 100``
        where ``limit = max_timestep_limit`` and ``best = time_best_ratio *
        limit`` (default half the limit). There is no completion gate: a round
        that does not dispose all bombs only ends when the limit is reached, so
        its ``elapsed_timestep`` is at the limit and the time score is 0. A
        crashed round earns 0, same as the red team.
        """
        if has_crashed:
            return 0.0

        limit = float(self.max_timestep_limit)
        elapsed = float(elapsed_timestep)

        if limit <= 0:
            return 0.0

        best = self.time_best_ratio * limit
        if limit == best:
            return 100.0 if elapsed <= best else 0.0

        fraction = (limit - elapsed) / (limit - best)
        fraction = max(0.0, min(1.0, fraction))
        return 100.0 * fraction
