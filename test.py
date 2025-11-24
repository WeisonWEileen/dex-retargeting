import xml.etree.ElementTree as ET
from pathlib import Path
import argparse


def fix_urdf_inertia(
    input_path: str,
    output_path: str | None = None,
    min_inertia: float = 1e-3,
) -> None:
    """
    Fix URDF links whose inertia diagonal entries (ixx, iyy, izz) are <= 0.

    - input_path: path to original URDF
    - output_path: path to save fixed URDF (if None, suffix '_fixed' is added)
    - min_inertia: value to use when ixx/iyy/izz <= 0
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_name(
            input_path.stem + "_fixed" + input_path.suffix
        )
    else:
        output_path = Path(output_path)

    tree = ET.parse(input_path)
    root = tree.getroot()

    ns = ""  # no XML namespace in this URDF

    def find(tag: str):
        return root.findall(f".//{tag}")

    num_links_fixed = 0

    for link in find("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue

        inertia = inertial.find("inertia")
        if inertia is None:
            continue

        name = link.get("name", "<unnamed>")

        def get_attr(attr: str) -> float:
            val = inertia.get(attr, None)
            if val is None:
                return 0.0
            try:
                return float(val)
            except ValueError:
                return 0.0

        ixx = get_attr("ixx")
        iyy = get_attr("iyy")
        izz = get_attr("izz")

        # 只要有一个对角线 <= 0，就把三个都修成 min_inertia
        if ixx <= 0.0 or iyy <= 0.0 or izz <= 0.0:
            inertia.set("ixx", str(min_inertia))
            inertia.set("iyy", str(min_inertia))
            inertia.set("izz", str(min_inertia))
            num_links_fixed += 1
            print(
                f"[fix_urdf_inertia] Fixed link '{name}': "
                f"ixx={ixx}, iyy={iyy}, izz={izz} -> {min_inertia}"
            )

    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(
        f"[fix_urdf_inertia] Done. Fixed {num_links_fixed} link(s). "
        f"Saved to: {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix URDF inertia entries (ixx/iyy/izz) that are <= 0."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to original URDF file.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="Path to save fixed URDF file. "
        "If not provided, will append '_fixed' to the filename.",
    )
    parser.add_argument(
        "--min_inertia",
        type=float,
        default=1e-3,
        help="Value to use when ixx/iyy/izz <= 0 (default: 1e-3).",
    )

    args = parser.parse_args()
    fix_urdf_inertia(
        input_path=args.input_path,
        output_path=args.output_path,
        min_inertia=args.min_inertia,
    )


if __name__ == "__main__":
    main()
