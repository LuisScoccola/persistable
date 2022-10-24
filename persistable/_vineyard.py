import numpy as np

_TOL = 1e-08


class Vineyard:
    def __init__(
        self,
        parameters,
        persistence_diagrams,
    ):
        self._parameters = list(parameters)
        self._persistence_diagrams = [
            [[p[0], p[1]] for p in pd] for pd in persistence_diagrams
        ]

    def parameter_indices(self):
        return list(range(len(self._parameters)))

    def _vineyard_to_vines(self):
        times = self.parameter_indices()

        def _prominences(bd):
            bd = np.array(bd)
            if bd.shape[0] == 0:
                return np.array([])
            else:
                return np.sort(np.abs(bd[:, 0] - bd[:, 1]))[::-1]

        prominence_diagrams = [_prominences(pd) for pd in self._persistence_diagrams]

        num_vines = np.max([len(prom) for prom in prominence_diagrams])
        padded_prominence_diagrams = np.zeros((len(times), num_vines))
        for i in range(len(times)):
            padded_prominence_diagrams[
                i, : len(prominence_diagrams[i])
            ] = prominence_diagrams[i]
        return [(times, padded_prominence_diagrams[:, j]) for j in range(num_vines)]

    def _vine_parts(self, prominences, tol=_TOL):
        times = self.parameter_indices()
        parts = []
        current_vine_part = []
        current_time_part = []
        part_number = 0
        for i, _ in enumerate(times):
            if prominences[i] < tol:
                if len(current_vine_part) > 0:
                    # we have constructed a non-trivial vine part that has now ended
                    if part_number != 0:
                        # this is not the first vine part, so we prepend 0 to the vine and the previous time to the times
                        current_vine_part.insert(0, 0)
                        current_time_part.insert(0, times[i - len(current_vine_part)])
                    # finish the vine part with a 0 and the time with the current time
                    current_vine_part.append(0)
                    current_time_part.append(times[i])
                    ## we save the current vine part and start over
                    parts.append(
                        (np.array(current_vine_part), np.array(current_time_part))
                    )
                    part_number += 1
                    current_vine_part = []
                    current_time_part = []
                # else, we haven't constructed a non-trivial vine part, so we just keep going
            elif i == len(times) - 1:
                if part_number != 0:
                    # this is not the first vine part, so we prepend 0 to the vine and the previous time to the times
                    current_vine_part.insert(0, 0)
                    current_time_part.insert(0, times[i - len(current_vine_part)])
                # finish the vine part with its value and the time with the current time
                current_vine_part.append(prominences[i])
                current_time_part.append(times[i])
                # we save the final vine part and time
                parts.append((np.array(current_vine_part), np.array(current_time_part)))
            else:
                # we keep constructing the vine part, since the prominence is non-zero
                current_vine_part.append(prominences[i])
                current_time_part.append(times[i])
            part_number += 1
        return parts
