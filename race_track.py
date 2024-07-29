import pygame
import sys
import numpy as np
from enum import Enum
from on_policy_mc import OnPolicyMonteCarloRaceTrack

# Define colors
BLACK = (40, 40, 34)
WHITE = (255, 255, 255)
BACKGROUND = (240, 240, 240)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 1200
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("My Pygame Window")
font = pygame.font.Font(None, 36)


def flip_y(y):
    return height - (y + 50)


class SquareType(Enum):
    BOUNDARY = 0
    EMPTY = 1
    START = 2
    FINISH = 3

    def get_colour(square_type):
        colours = [BLACK, BACKGROUND, RED, GREEN]
        return colours[square_type.value]


class AgentStatus(Enum):
    ALIVE = 0
    DEAD = 1
    WON = 2


class SimulationState(Enum):
    BUILDING = 0
    PLAYING = 1
    LEARNING = 2


class TextDisplay:
    def __init__(self, position, text, font_size=32, color=(0, 0, 0)):
        pygame.font.init()
        self.position = position
        self.text = text
        self.font = pygame.font.Font(None, font_size)
        self.color = color
        self.surface = None
        self.rect = None
        self.render_text()

    def render_text(self):
        self.surface = self.font.render(self.text, True, self.color)
        self.rect = self.surface.get_rect()
        self.rect.topleft = self.position

    def update_text(self, new_text):
        self.text = new_text
        self.render_text()

    def draw(self, screen):
        screen.blit(self.surface, self.rect)


class TrackSquare():

    def __init__(self, screen: pygame.Surface, state_index: tuple, type: SquareType, rect: tuple, ):
        self.state_index = state_index
        self.rect = pygame.Rect(rect)
        self.screen = screen
        self.type = type
        self.was_previously_pressed = False

    def does_collide(self, collision_position):
        return self.rect.collidepoint(collision_position)

    def rotate_type(self, direction=1):
        self.type = SquareType((self.type.value + direction) % len(SquareType))

    def draw(self):
        pygame.draw.rect(
            self.screen, SquareType.get_colour(self.type), self.rect)

    def get_x_and_y(self):
        return (self.rect.x, self.rect.y)


class Agent():

    def __init__(self, screen: pygame.Surface, start_state: tuple, track_squares: list[list[TrackSquare]]):
        self.screen = screen
        self.state = start_state
        self.velocity = (0, 0)
        self.max_velocity = 5
        self.width = 20
        self.height = 20
        self.track_squares = track_squares
        self.randomGenerator = np.random.default_rng()

    def get_pos(self):
        row, col = self.state
        row = min(row, len(self.track_squares)-1)
        col = min(col, len(self.track_squares[0])-1)
        state_square: TrackSquare = self.track_squares[row][col]
        x, y = state_square.get_x_and_y()
        x_coord = x + (state_square.rect.width - self.width)/2
        y_coord = y + (state_square.rect.width - self.height)/2

        return (x_coord, y_coord)

    def get_starting_positions(self):
        return [ts.state_index for row in self.track_squares for ts in row if ts.type == SquareType.START]

    def spawn(self):

        starting_states = self.get_starting_positions()
        if starting_states:
            (row, col) = self.randomGenerator.choice(starting_states, 1)[0]
            # Make sure to not use np.int64 type from choice returns
            random_start = (row.item(), col.item())
            self.set_state(random_start)
        else:
            self.set_state((0, 0))
        self.velocity = (0, 0)

    def set_state(self, state):
        (row, col) = state
        row = min(row, len(self.track_squares)-1)
        col = min(col, len(self.track_squares[0])-1)
        self.state = state

    def draw(self):
        x, y = self.get_pos()
        pygame.draw.rect(self.screen, BLUE,
                         (x, y, self.width, self.height))

    def set_change_in_velocity(self, delta_velocity):
        delta_vertical, delta_horizontal = delta_velocity

        # 1. Apply update
        current_vertical, current_horizontal = self.velocity

        current_vertical = max(current_vertical + delta_vertical, 0)
        current_horizontal = max(current_horizontal + delta_horizontal, 0)

        # 2. If both zero - make the one that was just changes into
        if (current_horizontal == 0 and current_vertical == 0):
            if delta_vertical < 0:
                current_horizontal = 1
            else:
                current_vertical = 1

        # 3. If velocity over max - cap it
        current_vertical = min(current_vertical, self.max_velocity)
        current_horizontal = min(current_horizontal, self.max_velocity)

        self.velocity = (current_vertical, current_horizontal)

    def move(self):
        row, col = self.state
        row += self.velocity[0]
        col += self.velocity[1]
        row = min(row, len(self.track_squares)-1)
        col = min(col, len(self.track_squares[0])-1)
        self.state = (row, col)

    def get_agent_status(self):
        row_velocity = self.velocity[0]
        col_velocity = self.velocity[1]
        start_row = self.state[0]
        start_col = self.state[1]

        for row in range(start_row, start_row+row_velocity+1):
            for col in range(start_col, start_col+col_velocity+1):
                if row >= len(self.track_squares) or col >= len(self.track_squares[0]):
                    continue
                if self.track_squares[row][col].type == SquareType.FINISH:
                    return AgentStatus.WON
                elif self.track_squares[row][col].type == SquareType.BOUNDARY:
                    return AgentStatus.DEAD

        return AgentStatus.ALIVE


class Button():
    def __init__(self, screen: pygame.Surface, position: tuple, callback, text, size: tuple):
        self.screen = screen
        self.position = position
        self.size = size
        self.callback = callback
        self.text = text
        self.rect = pygame.Rect(self.position, self.size)

    def draw(self):
        pygame.draw.rect(self.screen, WHITE, self.rect)
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect()
        text_rect.center = self.rect.center
        self.screen.blit(text_surface, text_rect)

    def does_collide(self, collision_position):
        return self.rect.collidepoint(collision_position)


# Create Squares for the track
border = 1
size = 30

cols = round((width)/(size+2*border))
rows = cols
track_squares: list[list[TrackSquare]] = []

step_size = border + size

y_pos = 0
x_pos = 0
index = (0, 0)
for _ in range(rows):
    row_of_track_squares = []
    for _ in range(cols):
        start_x = x_pos + border
        start_y = y_pos + border
        end_y = start_y + size
        end_x = start_x + size
        track_square = TrackSquare(
            screen, index, SquareType.BOUNDARY, (start_x, flip_y(start_y), size, size))

        x_pos += step_size
        row, col = index
        index = (row, col+1)
        row_of_track_squares.append(track_square)
    x_pos = 0
    y_pos += step_size
    row, col = index
    index = (row+1, 0)
    track_squares.append(row_of_track_squares)


# Create Agent for moving
agent = Agent(screen, (3, 4), track_squares)

# Create Button For Moving into PLAYING state
play_button = Button(screen, (50, 50), lambda _: _, "Play Mode", (200, 100))
simulate_button = Button(screen, (50, 50), lambda _: _,
                         "Simulation Mode", (200, 100))
toggle_policy_button = Button(screen, (25, 25), lambda _: _,
                              "Toggle Policy", (200, 100))

# Create count for episodes during simulation
episode_count_text = TextDisplay((300, 50), "Episode Count: 0")

# Create count for wins during simulation
win_count_text = TextDisplay((300, 100), "Win Count: 0")

# Create a clock object to control the frame rate
clock = pygame.time.Clock()

# Set Global State
simulation_state = SimulationState.BUILDING


def check_track_square_collision(collision_position, direction):
    for row in track_squares:
        for square in row:
            if square.does_collide(collision_position) and not square.was_previously_pressed:
                square.rotate_type(direction)
                square.was_previously_pressed = True


def check_play_button_collision(collision_position):
    global simulation_state
    if play_button.does_collide(collision_position):
        simulation_state = SimulationState.PLAYING


def check_simulation_button_collision(collision_position):
    global simulation_state
    if simulate_button.does_collide(collision_position):
        simulation_state = SimulationState.LEARNING


def check_toggle_policy_button_collision(collision_position):
    global is_learning
    if toggle_policy_button.does_collide(collision_position):
        is_learning ^= 1
        if is_learning:
            toggle_policy_button.text = "Is Learning"
        else:
            toggle_policy_button.text = "Not Learning"


def reset_was_previously_pressed():
    for row in track_squares:
        for square in row:
            square.was_previously_pressed = False


def building_racetrack():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif pygame.mouse.get_pressed()[0]:
            cursor_pos = pygame.mouse.get_pos()
            check_track_square_collision(cursor_pos, 1)
            check_play_button_collision(cursor_pos)
        elif pygame.mouse.get_pressed()[2]:
            cursor_pos = pygame.mouse.get_pos()
            check_track_square_collision(cursor_pos, -1)
        elif event.type == pygame.MOUSEBUTTONUP:
            reset_was_previously_pressed()

    play_button.draw()


def reset_agent(agent: Agent):
    agent.spawn()


def playing_racetrack(agent: Agent):

    agent_status = agent.get_agent_status()

    if agent_status == AgentStatus.DEAD or agent_status == AgentStatus.WON:
        print(agent_status.name)
        reset_agent(agent)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif pygame.mouse.get_pressed()[0]:
            cursor_pos = pygame.mouse.get_pos()
            check_simulation_button_collision(cursor_pos)

    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        agent.set_change_in_velocity((1, 0))
    elif keys[pygame.K_DOWN]:
        agent.set_change_in_velocity((-1, 0))
    elif keys[pygame.K_LEFT]:
        agent.set_change_in_velocity((0, -1))
    elif keys[pygame.K_RIGHT]:
        agent.set_change_in_velocity((0, 1))

    simulate_button.draw()
    agent.move()


# RL Algorithm to use
rl_algo = OnPolicyMonteCarloRaceTrack(environment_states=track_squares)

# Global Episode to track
episode = []
episode_count = 0
is_learning = True
wins = 0
loses = 0


def learning_racetrack():

    global episode_count
    global agent
    global rl_algo
    global episode
    global is_learning
    global wins
    global loses

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif pygame.mouse.get_pressed()[0]:
            cursor_pos = pygame.mouse.get_pos()
            check_toggle_policy_button_collision(cursor_pos)

    state = agent.state + agent.velocity
    action_index, action = rl_algo.policy(state, explore=is_learning)
    agent.set_change_in_velocity(action)
    agent.move()

    reward = -1  # Default reward for each step

    if agent.get_agent_status() == AgentStatus.WON:
        reward = 100  # Larger positive reward for winning
        episode.append((state, action, reward))
        if is_learning:
            rl_algo.every_visit_update(episode)
        episode = []
        episode_count += 1
        wins += 1
        episode_count_text.update_text(f"Episode count: {episode_count}")
        win_count_text.update_text(f"Win Count: {wins}")
        reset_agent(agent)
    elif agent.get_agent_status() == AgentStatus.DEAD:
        reward = -100  # Larger negative reward for crashing
        episode.append((state, action, reward))
        if is_learning:
            rl_algo.every_visit_update(episode)
        episode = []
        episode_count += 1
        loses += 1
        episode_count_text.update_text(f"Episode count: {episode_count}")
        reset_agent(agent)
    else:
        episode.append((state, action, reward))

    # rl_algo.epsilon = min(loses / (max(wins, 1) + loses), 0.5)

    episode_count_text.draw(screen)
    win_count_text.draw(screen)
    toggle_policy_button.draw()


# Main game loop
while True:
    # Fill the screen
    screen.fill(BACKGROUND)

    match simulation_state:
        case SimulationState.BUILDING:
            building_racetrack()
        case SimulationState.PLAYING:
            playing_racetrack(agent)
        case SimulationState.LEARNING:
            learning_racetrack()

    for row in track_squares:
        for square in row:
            square.draw()

    agent.draw()

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(240)
