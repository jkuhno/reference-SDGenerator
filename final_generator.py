import omni.replicator.core as rep
import io
import asyncio
import json
import numpy as np
import datetime as dt

from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.syntheticdata import SyntheticData


"""
@ Jani Kuhno, 2024

To run headless, see the tutorial: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/headless_example.html

Default logging directory on Windows: C:/Users/YOURUSER/.nvidia-omniverse/logs/Kit/Code/2022.3/

Default directory for coding samples:
C:\Users\YOURUSER\AppData\Local\ov\pkg\code-2023.1.1\extscache\omni.replicator.core-1.9.8+105.0.wx64.r.cp310

"""
__version__ = "1.0.0"

# START HERE!
# Use False when testing, only set to True when all is good, will run for an hour

isProduction = False
if isProduction:
    #final run
    cfg = {
          "output": "your_dataset_output_path",
          "colors": False,
          "frames": 10000,
          "subframes": 25,
          "format": "jpeg",
          "threads": 4,
          "queue": 1000
    }
else:
    #testing
    cfg = {
          "output": "your_testing_output_path",
          "colors": True,
          "frames": 10,
          "subframes": 25,
          "format": "png",
          "threads": 4,
          "queue": 1000
    }


 # Paths to usds, Omniverse Nucleus is recommended for collaboration.
 # Examples, if 'assets' folder in GitHub is put in Nucleus locally 
PROPS = 'omniverse://localhost/assets/POI/props/'
SCENE = 'omniverse://localhost/assets/scene/segmentation_background.usd'
ROUTER = 'omniverse://localhost/assets/scene/router.usd'

tables_dict = {
    'TABLE1' : 'omniverse://localhost/assets/POI/tables/table_1.usd',
    'TABLE2' : 'omniverse://localhost/assets/POI/tables/table_real.usd',
    'TABLE3' : 'omniverse://localhost/assets/POI/tables/table_2.usd',
    'TABLE4' : 'omniverse://localhost/assets/POI/tables/table_real.usd',
    'TABLE5' : 'omniverse://localhost/assets/POI/tables/table_3.usd',
    'TABLE6' : 'omniverse://localhost/assets/POI/tables/table_real.usd',
    }


# IMPORTANT: Always check if this dict has every class you want to label
# The semantic filter only allows to annotate classes in this dict, however check the test JSON if problems arise
# Make sure class id's are aligned with hand labelled samples for accurate IoU's, especially BACKGROUND and UNLABELLED
# BACKGROUND= pixels of every object in the scene without semantic class, or class ignored by semantic filter
# UNLABELLED = pixels of the void, just in case
classDict = {
    'BACKGROUND': 0,
    'props': 1,
    'table': 2,
    'UNLABELLED': 4
}
            

# Generate the semantic filter predicate string from keys of classDict dict
predicate = 'class:'
for idx, classes in enumerate(classDict.keys()):
    if idx == 0:
        predicate = predicate + classes
    else:    
        predicate = predicate +  '|' + classes


# Set global semantic filter predicate
SyntheticData.Get().set_instance_mapping_semantic_filter(predicate)

# Maximum thread count offered to the asyncronous encoding and writing to disk
# Default: 4
rep.settings.carb_settings("/omni/replicator/backend/writeThreads", cfg["threads"])

# Limit the queue from renderer to encoding and writing, if running out of system memory
# Default: 1000
rep.settings.carb_settings("/omni/replicator/backend/queueSize", cfg["queue"])


# Modified from the replicator.core BasicWriter
class CustomWriter(Writer):
    def __init__(self, output_dir: str,
                 classDict,
                 colorize_semantic_segmentation: bool = True,
                 image_format: str = "png",
                 isProduction: bool = False):
        self._frame_id = 0
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self.annotators = []
        self.colorize_semantic_segmentation = colorize_semantic_segmentation
        self.image_format = image_format
        self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        self.annotators.append(
            AnnotatorRegistry.get_annotator(
                "semantic_segmentation", init_params={"colorize": colorize_semantic_segmentation}
            )
        )

        self.CUSTOM_LABELS = classDict
        self.version = __version__

    # modified from BasicWriter
    def write(self, data):
        self._write_rgb(data, "rgb")
        self._write_segmentation(data, "semantic_segmentation")
        self._frame_id += 1

    def _write_rgb(self, data, annotator: str):
        # Save the rgb data under the correct path
        rgb_file_path = f"rgb_{self._frame_id}.{self.image_format}"
        self.backend.write_image(rgb_file_path, data[annotator])

    def _write_segmentation(self, data: dict, annotator: str):
        semantic_seg_data = data[annotator]["data"]
        id_to_labels = data[annotator]["info"]["idToLabels"]

        height, width = semantic_seg_data.shape[:2]

        file_path = (
            f"semantic_segmentation_{self._frame_id}.png"
        )
        if self.colorize_semantic_segmentation:
            semantic_seg_data = semantic_seg_data.view(np.uint8).reshape(height, width, -1)
            self.backend.write_image(file_path, semantic_seg_data)

        else:
            semantic_seg_data_labels = self.seg_data_as_labels(
                semantic_seg_data, id_to_labels, mapping=self.CUSTOM_LABELS
            )
            self.backend.write_image(file_path, semantic_seg_data_labels)


        # Produce .JSON files that display class id's from both annotator and custom forced dict
        # only when testing, because every production run should be first tested
        # NOTE the custom forced id's are not deployed in pictures when running colorized
        if isProduction == False:
            file_path = f"semantic_segmentation_labels_jason_{self._frame_id}.json"
            buf = io.BytesIO()
            buf.write(json.dumps("From annotator: " + str({str(k): v for k, v in id_to_labels.items()}) + '\\n' +
                                 "From custom: " + str({str(k): v for k, v in self.CUSTOM_LABELS.items()})
            ).encode())
            self.backend.write_blob(file_path, buf.getvalue())
                    

    # Modified from omni.replicator.core.tools.colorize_segmentation
    # Same as that but instead of color mapping in shape (width, height, 4)
    # it maps custom int labels for seg mask in shape (width, height)
    def seg_data_as_labels(self, data, labels, mapping):
        unique_ids = np.unique(data)
        seg_as_labels = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
        for i, _id in enumerate(unique_ids):
            obj_label = [*labels[str(_id)].values()][0].lower()
            if obj_label in mapping:
                seg_as_labels[data == _id] = mapping[obj_label]

        return seg_as_labels

    def on_final_frame(self):
        self.backend.sync_pending_paths()


#End of the writer class
##########################################

# Register new writer
WriterRegistry.register(CustomWriter)


#open a new layer in order to not mess up the usd
with rep.new_layer():

    # randomize lights, sphere light instead of distant or dome in order to create shadows
    # position relative to the table, xyz
    # carb_settings modifies the default lighting, ensures atleast a little ambient lighting
    def sphere_lights():
        rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity",
                                   rep.distribution.sequence([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
                                   )
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.choice([3000, 3500, 4000, 4500, 5000, 5500, 6000]),
            intensity=rep.distribution.uniform(20000, 70000),
            scale=200,
            position=(300, 700, -350),
            count=1
        )
        return lights.node


    # background, objects in the scene are annotated in OV Code using semantic schema editor
    # SEAMNTIC CLASSES: BACKGROUND, UNLABELLED, carpet, crutches, pc, drawer, trashbin
    scene = rep.create.from_usd(SCENE)
    with scene:
         rep.modify.pose(position=(580, 0 ,-700), scale=2)

    # Add a router object to the corner of the table
    router = rep.create.from_usd(ROUTER, semantics=[('class', 'props')])
    with router:
        rep.modify.pose(position=(-168, 192, 70),
                        rotation=(-366, -19, -37),
                        scale=(1.5, 1, 1)
                )


    # create a plane to sample props on in randomize_props
    # position(x,y,z) y is around the height of the tables,
    # scale(x,y,z) is a little bit smaller than the dimensions of tables
    plane_samp = rep.create.plane(position=(0, 225, 0), scale=(2, 1, 1.5), visible=False)
    

    # camera, name for debugging in OV Code
    camera = rep.create.camera(focus_distance=100, look_at=plane_samp, name="main_camera")


    # create a renderer, resolution is defined here. TODO: add the reso tuple to cfg dict
    rp = rep.create.render_product(camera, (512, 512))


    # Create the tables for variation
    # the list will hold the references to table prims
    # SEMANTIC CLASSES: table
    tables = []
    for x in tables_dict:
        table_usd = rep.create.from_usd(tables_dict[x], semantics=[('class', 'table')])
        tables.append(table_usd)


    # function for scattering the instantiated prop prims on to the plane_samp, hidden inside the tables
    # SEMANTIC CLASSES: props
    def randomize_props():
        props = rep.randomizer.instantiate(
            rep.utils.get_usd_files(PROPS), 
            size=3, 
            with_replacements=False, 
            mode='scene_instance'
            )

        # scatter on a plane
        with props:
            rep.modify.pose(rotation=(0, 180, 0))
            rep.modify.semantics([("class", 'props')])
            rep.randomizer.scatter_2d(plane_samp)
        return props.node


    # Visibility distribution sequence for the table randomization
    # Produces viz_matrix, which aligns with the number of tables in tables[], for example with 3 tables
    # [[True, False, False],
    #  [False, True, False],
    #  [False, False, True]]
    #
    one_sequence = [False] * len(tables)
    router_sequence = [False] * len(tables)
    viz_matrix = []
    for x in range (len(tables)):
        arr = one_sequence.copy()
        arr[x] = True
        viz_matrix.append(arr)


    # Visibility distribution for the router to only appear with the real table
    for i, (k, v) in enumerate(tables_dict.items()):
        if v == 'omniverse://localhost/Library/tables/table_real.usd':
            router_sequence[i] = True


    # Register defined randomization functions to the randomizer
    rep.randomizer.register(sphere_lights)
    rep.randomizer.register(randomize_props)

    # Call the randomizer on each frame
    with rep.trigger.on_frame(num_frames=cfg["frames"], rt_subframes=cfg["subframes"]):
        for idx, table in enumerate(tables):
            with table:
                rep.modify.visibility(rep.distribution.sequence(viz_matrix[idx]))

        with router:
            rep.modify.visibility(rep.distribution.sequence(router_sequence)) 

        with camera:
            rep.modify.pose(position=rep.distribution.uniform((0, 300, -600), (400, 700, -300)), look_at=plane_samp)

        rep.randomizer.randomize_props()
        rep.randomizer.sphere_lights()


# Initialize and attach writer
writer = rep.WriterRegistry.get("CustomWriter")
writer.initialize(output_dir=cfg["output"],
                  classDict=classDict,
                  colorize_semantic_segmentation=cfg["colors"],
                  image_format=cfg["format"],
                  isProduction=isProduction)
writer.attach([rp])


async def run_generator():
    await rep.orchestrator.run_async()
    print(f"Start: {dt.datetime.now()}")
    await rep.orchestrator.run_until_complete_async()
    print(f"Finish: {dt.datetime.now()}")

asyncio.ensure_future(run_generator())
