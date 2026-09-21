"""Geometry matrices for ordered Molecules loaded by the original reader.

Notes for downstream exports:
  *_summary files: summaries for later collective-variable screening and
  selection.
  *_values files: all individual values for each Molecule, for the atom
  combinations included in the profile. Rows represent Molecules/geometries;
  columns identify the individual distances, angles or dihedrals.
"""

import os

import numpy as np

from chemsmart.io.molecules.structure import Molecule


def _geometry_summary(values, circular=False):
    """
    Torsion endpoint change is the shortest signed difference in [-180,180).
    """
    summary = np.full((values.shape[1], 7), np.nan)
    counts = np.isfinite(values).sum(axis=0)
    for column in range(values.shape[1]):
        series = values[:, column]
        first, last = series[0], series[-1]
        change = last - first
        if circular:
            change = (change + 180.0) % 360.0 - 180.0
        summary[column, :4] = first, last, change, abs(change)
        finite = series[np.isfinite(series)]
        if not len(finite):
            continue
        if circular:
            ordered = np.sort(finite % 360.0)
            gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
            gap = int(np.argmax(gaps))
            lower = ordered[(gap + 1) % len(ordered)]
            lower = (lower + 180.0) % 360.0 - 180.0
            span = max(0.0, 360.0 - gaps[gap])
            summary[column, 4:] = lower, lower + span, span
        else:
            summary[column, 4:] = finite.min(), finite.max(), np.ptp(finite)
    columns = (
        "start",
        "end",
        "signed_change",
        "absolute_change",
        "arc_start" if circular else "minimum",
        "arc_end" if circular else "maximum",
        "circular_range" if circular else "range",
    )
    return summary, columns, counts


def _angular_profile(molecules, property_name, order):
    tables = [getattr(molecule, property_name) for molecule in molecules]
    detected = [
        {tuple(int(i) for i in row[:order]): row[-1] for row in table}
        for table in tables
    ]
    combinations = sorted(set().union(*(set(rows) for rows in detected)))
    indices = np.asarray(combinations, dtype=int).reshape(-1, order)
    values = np.empty((len(molecules), len(indices)), dtype=float)
    connected = np.zeros(values.shape, dtype=bool)
    for frame, (molecule, rows) in enumerate(zip(molecules, detected)):
        missing = []
        for column, atoms in enumerate(combinations):
            if atoms in rows:
                values[frame, column] = rows[atoms]
                connected[frame, column] = True
            else:
                missing.append(column)
        if missing:
            values[frame, missing] = molecule.geometry_table(indices[missing])[
                :, -1
            ]
    summary, columns, counts = _geometry_summary(values, circular=order == 4)
    return {
        "atom_indices": indices,
        "values_matrix": values,
        "connected_matrix": connected,
        "summary_matrix": summary,
        "summary_columns": columns,
        "valid_counts": counts,
        "units": "degree",
    }


def geometry_profile(molecules):
    """Collect distance/angle/dihedral properties across ordered geometries.
    """
    molecules = tuple(molecules)
    if not molecules:
        raise ValueError("Provide at least one geometry.")
    if not all(isinstance(molecule, Molecule) for molecule in molecules):
        raise TypeError("All geometries must be Molecule objects.")
    symbols = list(molecules[0].symbols)
    for molecule in molecules:
        if list(molecule.symbols) != symbols:
            raise ValueError("Geometries have different element sequences.")
        molecule._validate_matrix_positions()
    first, second = np.triu_indices(len(symbols), k=1)
    indices = np.column_stack((first + 1, second + 1))
    values = np.stack(
        [molecule.distances_matrix[first, second] for molecule in molecules]
    )
    lookup = {tuple(pair): column for column, pair in enumerate(indices)}
    connected = np.zeros(values.shape, dtype=bool)
    for frame, molecule in enumerate(molecules):
        for i, j in molecule.to_graph().edges:
            connected[frame, lookup[tuple(sorted((i + 1, j + 1)))]] = True
    summary, columns, counts = _geometry_summary(values)
    distances = {
        "atom_indices": indices,
        "values_matrix": values,
        "connected_matrix": connected,
        "selected_mask": connected.any(axis=0),
        "summary_matrix": summary,
        "summary_columns": columns,
        "valid_counts": counts,
        "units": "angstrom",
    }
    return {
        "molecules": molecules,
        "distances": distances,
        "angles": _angular_profile(molecules, "angles_matrix", 3),
        "dihedrals": _angular_profile(molecules, "dihedrals_matrix", 4),
        "connectivity_method": "Molecule.to_graph(default settings)",
    }


def geometry_profile_from_file(filepath):
    """Use the original Molecule.from_filepath reader for all geometries.
    """
    filepath = os.path.abspath(filepath)
    molecules = Molecule.from_filepath(filepath, index=":", return_list=True)
    if not molecules:
        raise ValueError("No geometries were returned by from_filepath.")
    result = geometry_profile(molecules)
    result["source_file"] = filepath
    result["reader"] = "Molecule.from_filepath"
    result["geometry_index_definition"] = "1-based position in reader output"
    return result


def atom_maximum_changes(profile, connected_distances=True):
    """Summarise largest absolute endpoint changes involving each atom.
    """
    categories = ("distances", "angles", "dihedrals")
    symbols = list(profile["molecules"][0].symbols)
    labels = tuple(f"{symbol}{i}" for i, symbol in enumerate(symbols, 1))
    values = np.full((len(symbols), 3), np.nan)
    notes = np.empty((len(symbols), 3), dtype=object)
    candidate_counts = np.zeros(values.shape, dtype=int)
    undefined_counts = np.zeros(values.shape, dtype=int)
    winner_columns = {}
    for category_column, category in enumerate(categories):
        data = profile[category]
        indices = data["atom_indices"]
        changes = data["summary_matrix"][
            :, data["summary_columns"].index("absolute_change")
        ]
        eligible = np.ones(len(indices), dtype=bool)
        if category == "distances" and connected_distances:
            eligible &= data["selected_mask"]
        winners_by_atom = []
        for atom in range(1, len(symbols) + 1):
            candidates = eligible & (indices == atom).any(axis=1)
            finite = candidates & np.isfinite(changes)
            row = atom - 1
            candidate_counts[row, category_column] = candidates.sum()
            undefined_counts[row, category_column] = (
                candidates & ~finite
            ).sum()
            if not finite.any():
                notes[row, category_column] = (
                    "Undefined endpoint changes"
                    if candidates.any()
                    else "No eligible coordinate"
                )
                winners_by_atom.append(())
                continue
            maximum = changes[finite].max()
            winners = sorted(
                np.flatnonzero(finite & (changes == maximum)),
                key=lambda column: tuple(indices[column]),
            )
            values[row, category_column] = maximum
            notes[row, category_column] = "; ".join(
                "-".join(labels[i - 1] for i in indices[column])
                for column in winners
            )
            winners_by_atom.append(tuple(int(i) for i in winners))
        winner_columns[category] = tuple(winners_by_atom)
    return {
        "atom_labels": labels,
        "categories": categories,
        "units": ("angstrom", "degree", "degree"),
        "values_matrix": values,
        "notes_matrix": notes,
        "winner_columns": winner_columns,
        "candidate_counts": candidate_counts,
        "undefined_counts": undefined_counts,
        "distance_selection": (
            "connected_any" if connected_distances else "all"
        ),
    }
