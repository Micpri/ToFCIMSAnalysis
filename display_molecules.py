# Write parser
import svgutils.compose as svgc
import svgutils.transform as svgt

def svg_element_node_string(species, name_index, im_path, scale_inc, img_scaler, fs, pannel_i):

    """
    returns string to be parsed that creates an image of a species with its name underneath and provides all necesarry positional offsets
    re. other images that will be generated for that row.
    """

    return """
    svgc.SVG("{2}"+plot_dict["{0}"][{1}]+".svg").scale({4}).move( ({1} + 1) * {3} - ({3} / 4), {6} * {3}),
    svgt.TextElement(({1} + 1) * {3}, ({6} + 1) * {3}, str({1})+". "+plot_dict["{0}"][{1}], size = {5}),""".format(species, name_index, im_path, scale_inc, img_scaler, fs, pannel_i)


def svg_pannel_string(panels, species, scale_inc, fs, index):
    
    """
    returns string to be parsed that creates a pannel of a series of images created by svg_element_node_string and provides all necesarry positional offsets
    re. other pannels that will be generated for that figure.
    """
    
    return """svgc.Panel(
    svgt.TextElement({1}/10, (({3} + 1) * {1}) + ({1}/2) - {1}, str({3})+'. {0}', size={2}), 
    %s
    ),""".format(species, scale_inc, fs, index) % panels[species]


def build_svg_panels(plot_dict, im_path, scale_inc, img_scaler, fs):
    
    """panels holds the different strings that represent the svg panel i.e. all species columns in a row, indexed by the species name (i.e. row)."""
    
    panels = collections.OrderedDict({})
    for pannel_i, species in enumerate(plot_dict.keys()):
        panels[species] = []  
        
        for species_i, name in enumerate(plot_dict[species]):
            panels[species].append(svg_element_node_string(species, species_i, im_path, scale_inc, img_scaler, fs, pannel_i))
        
        panels[species] = "\n".join(panels[species])[:-1]
    
    return panels


def build_svg_figbody_from_panels(panel, scale_inc, fs):
    
    """panels holds the different strings that represent the species columns of a row, indexed by the species name (i.e. row).
    These need putting into the figure."""

    figure_body = []
    
    for i, species in enumerate(panels):
        figure_body.append(svg_pannel_string(panels, species, scale_inc, fs, i))   
    figure_body = "".join(figure_body)   

    return figure_body
    

def write_svg_figure(figure_body, fscoef, rows, cols, group, fs):
    
    """Wraps figure body text with svgc figure and saves to file."""
    
    return """svgc.Figure(
    str({0}*{2}*5)+"cm", str({0}*{1})+"cm",
    svgt.TextElement(20, 70,'{3}', size={4}+10), 
    %s
    )""".format(fscoef, rows, cols, group, fs)  % figure_body


def plot_dict_required(df, group):
    
    plot_dict = collections.OrderedDict({})
    for ion in df["groups"][group]:
        try:
            plot_dict[ion] = df["pl"][df["pl"]["ion"] == ion].tag.values[0].split(", ")
        except IndexError:
            plot_dict[ion] = ['question_mark']
        
    return plot_dict


def rows_required(plot_dict):
    return len(plot_dict.keys())


def cols_required(plot_dict):
    return len(max(plot_dict.values(), key=lambda coll: len(coll)))


# Global variables for parser
save_path = "/home/mbexkmp3/Documents/benzene_paper/images/"
im_path = save_path+"benzene_oxidation_products/" # path to svgs
scale_inc = 300 # distance between images
fs = 14 # fontsize
img_scaler = 1.3 # scale image 
fscoef = scale_inc / 33. # coeficient used to size the figure relative to number of rows and cols


# variables specific to group
group = 'nSS_ID_HOM_I'
plot_dict = plot_dict_required("NOx0ppb", group)
rows = rows_required(plot_dict)
cols = cols_required(plot_dict)
save_name = save_path+group+'.svg'

# MAIN
panels = build_svg_panels(plot_dict, im_path, scale_inc, img_scaler, fs)
figure_body = build_svg_figbody_from_panels(panels, scale_inc, fs) # ok up to here!
svg_string = write_svg_figure(figure_body, fscoef, rows, cols, group, fs)