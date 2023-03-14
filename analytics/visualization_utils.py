import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation


def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1)

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def fig_canvas_image(fig):
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
    """Compute a color map array of shape [num_agents, 4]."""
    colors = cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors


def get_viewport(states, masks):
    valid_states = states[masks]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_x, center_y, width


def visualize_frame(ego_state, ego_mask, agents_state, agents_mask, roadgraph,
                    center_x, center_y, width, color_map, title, size_pixels=1000):
    fig, ax = create_figure_and_axes(size_pixels)

    roadgraph_points = roadgraph[:, :2].transpose()
    ax.plot(roadgraph_points[0, :], roadgraph_points[1, :], 'k.', alpha=1, ms=2)

    agents_masked_x = agents_state[:, 0][agents_mask]
    agents_masked_y = agents_state[:, 1][agents_mask]

    ax.scatter(agents_masked_x, agents_masked_y, marker='o', linewidths=3, color='b')

    # plot ego
    if ego_mask:
        ax.scatter(ego_state[0], ego_state[1], marker='*', linewidths=3, color='r')

    # Title.
    ax.set_title(title)

    # Set axes.  Should be at least 10m on a side and cover 160% of agents.
    size = max(10, width * 1.0)
    ax.axis([-size / 2 + center_x,
             size / 2 + center_x,
             -size / 2 + center_y,
             size / 2 + center_y])
    ax.set_aspect('equal')

    image = fig_canvas_image(fig)
    plt.close(fig)
    return image


def visualize_scenario(data, size_pixels=1000):
    ego_id: int = data.tracks[data.sdc_track_index].id
    ego_states = np.array([[state.center_x, state.center_y] for state in data.tracks[data.sdc_track_index].states])
    ego_masks = np.array([state.valid for state in data.tracks[data.sdc_track_index].states])
    # [num_agents, num_steps, 2]
    agent_states = np.array([[[state.center_x, state.center_y] for state in track.states]
                            for track in data.tracks if track.id != ego_id])
    # [num_agents, num_steps]
    agent_masks = np.array([[state.valid for state in track.states]
                            for track in data.tracks if track.id != ego_id])
    num_agents, num_steps, _ = agent_states.shape

    # road graph
    features = data.map_features
    lanes = [feature.lane for feature in features if feature.WhichOneof("feature_data") == "lane"]
    road_lines = [feature.road_line for feature in features if feature.WhichOneof("feature_data") == "road_line"]
    road_edges = [feature.road_edge for feature in features if feature.WhichOneof("feature_data") == "road_edge"]
    stop_signs = [feature.stop_sign for feature in features if feature.WhichOneof("feature_data") == "stop_sign"]
    crosswalks = [feature.crosswalk for feature in features if feature.WhichOneof("feature_data") == "crosswalk"]
    speed_bumps = [feature.speed_bump for feature in features if feature.WhichOneof("feature_data") == "speed_bump"]

    lane_points = np.array([[point.x, point.y] for lane in lanes for point in lane.polyline])
    road_line_points = np.array([[point.x, point.y] for road_line in road_lines for point in road_line.polyline])
    road_edge_points = np.array([[point.x, point.y] for road_edge in road_edges for point in road_edge.polyline])

    roadgraph = np.vstack((road_line_points, road_edge_points))

    color_map = get_colormap(num_agents)
    center_x, center_y, width = get_viewport(agent_states, agent_masks)

    images = []
    for i, (agent_state, agent_mask) in enumerate(zip(np.split(agent_states, num_steps, 1),
                                                      np.split(agent_masks, num_steps, 1))):
        image = visualize_frame(ego_states[i], ego_masks[i],
                                agent_state[:, 0], agent_mask[:, 0],
                                roadgraph,
                                center_x, center_y, width, color_map,
                                f"frame:{i}", size_pixels)
        images.append(image)

    return images


def create_animation(images):
    plt.ioff()
    fig, ax = plt.subplots()
    dpi = 100
    size_inches = 1000 / dpi
    fig.set_size_inches([size_inches, size_inches])
    plt.ion()

    def animate_func(idx):
        ax.imshow(images[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid('off')

    anim = animation.FuncAnimation(fig, animate_func, frames=len(images) // 2, interval=100)
    plt.close(fig)
    return anim
