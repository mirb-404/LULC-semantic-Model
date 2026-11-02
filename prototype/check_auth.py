"""Check that Google Earth Engine works for the prototype: login, project and data access.

Run from this folder:  python check_auth.py [--project <id>]
Exit code 0 if everything works, 1 otherwise.
"""
import argparse
import sys
from pathlib import Path

import ee

import gee

AUTH_HINT = ("Log in once:  earthengine authenticate\n"
             "  headless machine (Datalab):  earthengine authenticate --auth_mode=notebook\n"
             "  open the link, paste the code back, and do not tick 'read-only scopes'.")
PROJECT_HINT = ("The project cannot use Earth Engine (not registered, access expired or API disabled).\n"
                "Use a project registered for Earth Engine: --project <id>, EE_PROJECT=<id> or Project_ID in .env.")


def fail(step, err, hint):
    print(f"FAIL  {step}: {err}\n\n{hint}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default=gee.default_project(),
                        help="Earth Engine project (default: EE_PROJECT, .env Project_ID, or the built-in default)")
    project = parser.parse_args().project
    print(f"project: {project}")

    credentials = Path(ee.oauth.get_credentials_path())
    if not credentials.exists():
        fail("credentials", f"no file at {credentials}", AUTH_HINT)
    print(f"OK    credentials file: {credentials}")

    try:
        gee.initialize(project)
        ee.Number(1).getInfo()
    except Exception as err:
        text = str(err)
        if any(s in text for s in ("authorize", "credentials", "invalid_scope", "invalid_grant", "expired")
               ) and "access has expired" not in text:
            fail("login", text, AUTH_HINT)
        fail("project", text, PROJECT_HINT)
    print("OK    login and project")

    for name, collection in (("Sentinel-1 GRD", "COPERNICUS/S1_GRD"), ("Esri Annual LULC", gee.ESRI_COLLECTION)):
        try:
            ee.ImageCollection(collection).limit(1).size().getInfo()
        except Exception as err:
            fail(name, err, "The project works but cannot read this dataset.")
        print(f"OK    data access: {name}")
    print("\nEarth Engine is ready for the prototype.")


if __name__ == "__main__":
    main()
