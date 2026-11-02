import pygame
import soundfile as sf
import numpy as np
import sys
import time

def draw_rolling_waveform(surface, audio_data, color, width, height, current_sample, zoom_factor):
    """Extracts a slice of audio centered on the current sample and draws it."""
    surface.fill((20, 20, 20)) # Clear the background each frame
    
    half_width = width // 2
    start_sample = current_sample - (half_width * zoom_factor)
    end_sample = start_sample + (width * zoom_factor)
    
    # Calculate safe bounds to slice from the actual audio array
    src_start = max(0, start_sample)
    src_end = min(len(audio_data), end_sample)
    
    # Prepare a zero-padded window for the screen (handles audio start/end boundaries)
    window_data = np.zeros(width * zoom_factor, dtype=np.float32)
    
    dest_start = 0 if start_sample >= 0 else -start_sample
    dest_end = dest_start + (src_end - src_start)
    
    if src_start < src_end:
        window_data[dest_start:dest_end] = audio_data[src_start:src_end]
        
    # Group the audio into columns (1 pixel = 'zoom_factor' samples) and find peak amplitude
    reshaped = window_data.reshape(width, zoom_factor)
    peaks = np.max(np.abs(reshaped), axis=1)
    
    mid_y = height // 2
    for x in range(width):
        amp = int(peaks[x] * (height / 2))
        if amp > 0:
            pygame.draw.line(surface, color, (x, mid_y - amp), (x, mid_y + amp), 1)
        else:
            pygame.draw.line(surface, color, (x, mid_y), (x, mid_y), 1) # Silence dot

def rolling_ab_tester(input_file, output_file):
    pygame.init()
    pygame.mixer.init()

    WIDTH, HEIGHT = 1000, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Rolling A/B Waveform Comparison")
    font = pygame.font.Font(None, 36)

    try:
        print("Loading audio data for waveforms...")
        data_a, sr_a = sf.read(input_file, dtype='float32')
        data_b, sr_b = sf.read(output_file, dtype='float32')
        duration = len(data_a) / sr_a
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

    if len(data_a.shape) > 1: data_a = data_a[:, 0]
    if len(data_b.shape) > 1: data_b = data_b[:, 0]

    surface_a = pygame.Surface((WIDTH, HEIGHT // 2))
    surface_b = pygame.Surface((WIDTH, HEIGHT // 2))

    track_a = pygame.mixer.Sound(input_file)
    track_b = pygame.mixer.Sound(output_file)
    
    channel_a = track_a.play(loops=-1) 
    channel_b = track_b.play(loops=-1)

    listening_to_a = True
    channel_a.set_volume(1.0)
    channel_b.set_volume(0.0)

    is_paused = False
    playback_time = 0.0
    last_update = time.time()
    
    # Zoom settings: represents how many audio samples are compressed into 1 pixel
    zoom_factor = 128 

    running = True
    while running:
        current_time = time.time()
        if not is_paused:
            playback_time += (current_time - last_update)
            if playback_time > duration:
                playback_time %= duration
        last_update = current_time
        
        current_sample = int(playback_time * sr_a)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    listening_to_a = not listening_to_a
                    if listening_to_a:
                        channel_a.set_volume(1.0)
                        channel_b.set_volume(0.0)
                    else:
                        channel_a.set_volume(0.0)
                        channel_b.set_volume(1.0)
                elif event.key == pygame.K_p:
                    is_paused = not is_paused
                    if is_paused:
                        pygame.mixer.pause()
                    else:
                        pygame.mixer.unpause()
                        last_update = time.time()
                # Zoom Controls
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    zoom_factor = max(1, zoom_factor // 2) # Zoom in
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom_factor = min(8192, zoom_factor * 2) # Zoom out

        # Dynamically render the current viewable slice of the waveforms
        draw_rolling_waveform(surface_b, data_b, (100, 255, 100), WIDTH, HEIGHT // 2, current_sample, zoom_factor)
        draw_rolling_waveform(surface_a, data_a, (255, 100, 100), WIDTH, HEIGHT // 2, current_sample, zoom_factor)

        screen.fill((0, 0, 0))
        screen.blit(surface_b, (0, 0))
        screen.blit(surface_a, (0, HEIGHT // 2))
        
        # Draw static center Playhead
        pygame.draw.line(screen, (255, 255, 255), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)

        status_text = "PAUSED" if is_paused else "PLAYING"
        track_text = "Noisy Input (Bottom)" if listening_to_a else "Cleaned Output (Top)"
        
        text1 = font.render(f"Hearing: {track_text} | Zoom: {zoom_factor}x", True, (255, 255, 255))
        text2 = font.render(f"[{status_text}] SPACE: Toggle | P: Pause | +/-: Zoom", True, (200, 200, 200))
        
        pygame.draw.rect(screen, (0, 0, 0), (10, 10, text1.get_width() + 10, text1.get_height() + 5))
        pygame.draw.rect(screen, (0, 0, 0), (10, HEIGHT // 2 + 10, text2.get_width() + 10, text2.get_height() + 5))
        
        screen.blit(text1, (15, 15))
        screen.blit(text2, (15, HEIGHT // 2 + 15))

        pygame.display.flip()
        pygame.time.Clock().tick(60)

    pygame.quit()

if __name__ == "__main__":
    rolling_ab_tester("noisy_input.wav", "cleaned_output.wav")
