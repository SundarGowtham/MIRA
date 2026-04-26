from mp_api.client import MPRester
import dotenv
import os


dotenv.load_dotenv()

MP_API_KEY = os.environ.get("MP_API_KEY")

if not MP_API_KEY:
    raise ValueError("Missing required environment variable: MP_API_KEY")


with MPRester(MP_API_KEY) as mpr:
    # mpr.materials.

    
    docs = mpr.materials.summary.search(
        material_ids=["mp-11351"]
    )

#     # print(docs)

#     # Check synthesis endpoint
#     results = mpr.synthesis.search(
#         keywords=["solid-state"],
#         num_chunks=2
#     )
#     if results:
#         r = results[0]
#         print(type(r))
#         print(dir(r))
#         print()
#         print(r)


# with MPRester(MP_API_KEY) as mpr:
#     results = mpr.materials.synthesis.search(
#         synthesis_type=["solid-state"],
#         num_chunks=100
#     )
#     print(f"Total solid-state records: {len(results)}")
    
#     # Check how many have complete reaction data
#     complete = [r for r in results 
#                 if r.reaction and r.precursors and r.target]
#     print(f"Records with complete reaction data: {len(complete)}")

# from mace.calculators import mace_mp
# calc = mace_mp(model="medium", device="cpu")
# print("MACE ready")