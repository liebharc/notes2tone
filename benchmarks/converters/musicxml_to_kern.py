"""Convert MusicXML to **kern format.

This module implements a custom MusicXML to **kern converter using music21.
Since neither music21 nor verovio provide native MusicXML → **kern export,
we manually construct **kern syntax from parsed musical elements.

TODO: Check if there's a better library for this conversion
TODO: Handle more complex musical notation (articulations, dynamics, etc.)
"""

import logging
from music21 import converter, note, chord, meter, key, clef
from typing import Optional

logger = logging.getLogger(__name__)


def _pitch_to_kern(pitch) -> str:
    """Convert music21 pitch to **kern notation.
    
    **kern pitch notation:
    - Lowercase = octave 4 and above (c=C4, d=D4, cc=C5)
    - Uppercase = octave 3 and below (C=C3, D=D3, CC=C2)
    - # for sharp, - for flat
    """
    step = pitch.step.lower()
    octave = pitch.octave
    
    # Determine case based on octave (C4 = middle C = lowercase 'c')
    if octave >= 4:
        # Octave 4+: lowercase letters, repeat for higher octaves
        kern_pitch = step * (octave - 3)
    else:
        # Octave 3-: uppercase letters, repeat for lower octaves
        kern_pitch = step.upper() * (4 - octave)
    
    # Add accidentals
    if pitch.accidental:
        if pitch.accidental.name == 'sharp':
            kern_pitch += '#'
        elif pitch.accidental.name == 'flat':
            kern_pitch += '-'
        elif pitch.accidental.name == 'double-sharp':
            kern_pitch += '##'
        elif pitch.accidental.name == 'double-flat':
            kern_pitch += '--'
    
    return kern_pitch


def _duration_to_kern(duration) -> str:
    """Convert music21 duration to **kern duration."""
    # **kern uses reciprocal notation: 4 = quarter, 8 = eighth, etc.
    # Special cases: 0 = breve, 00 = long
    
    quarter_length = duration.quarterLength
    
    if quarter_length == 8.0:
        return '0'  # Breve
    elif quarter_length == 4.0:
        return '1'  # Whole note
    elif quarter_length == 2.0:
        return '2'  # Half note
    elif quarter_length == 1.0:
        return '4'  # Quarter note
    elif quarter_length == 0.5:
        return '8'  # Eighth note
    elif quarter_length == 0.25:
        return '16'  # Sixteenth note
    elif quarter_length == 0.125:
        return '32'  # 32nd note
    else:
        # For dotted notes, use base duration + dot
        if quarter_length == 3.0:
            return '2.'
        elif quarter_length == 1.5:
            return '4.'
        elif quarter_length == 0.75:
            return '8.'
        elif quarter_length == 0.375:
            return '16.'
        else:
            # Fallback: use quarter note
            return '4'


def convert_musicxml_to_kern(musicxml_content: str) -> str:
    """Convert MusicXML string to **kern format.
    
    This is a simplified converter focusing on pitch and rhythm.
    Complex features (articulations, dynamics, ornaments) are not fully supported.
    
    Args:
        musicxml_content: MusicXML content as string
        
    Returns:
        **kern format string
        
    Raises:
        ValueError: If conversion fails
    """
    try:
        # Parse MusicXML
        score = converter.parseData(musicxml_content, format='musicxml')
        
        if not score.parts:
            raise ValueError("No parts found in MusicXML")
        
        # Build parallel kern notation for all parts
        parts_data = []
        max_lines = 0
        
        for part in score.parts:
            part_lines = []
            
            # Header
            part_lines.append("**kern")
            
            # Add key signature if present
            key_sig = part.flatten().getElementsByClass(key.KeySignature)
            if key_sig:
                k = key_sig[0]
                if hasattr(k, 'sharps') and k.sharps != 0:
                    part_lines.append(f"*k[{k.sharps}#]")
                elif hasattr(k, 'flats') and getattr(k, 'flats', 0) != 0:
                    part_lines.append(f"*k[{k.flats}-]")
            
            # Add clef if present
            clef_obj = part.flatten().getElementsByClass(clef.Clef)
            if clef_obj:
                c = clef_obj[0]
                part_lines.append(f"*clef{c.sign}{c.line}")
            
            # Add time signature if present
            time_sig = part.flatten().getElementsByClass(meter.TimeSignature)
            if time_sig:
                ts = time_sig[0]
                part_lines.append(f"*M{ts.numerator}/{ts.denominator}")
            
            # Process notes and rests
            for element in part.flatten().notesAndRests:
                if isinstance(element, note.Note):
                    duration_str = _duration_to_kern(element.duration)
                    pitch_str = _pitch_to_kern(element.pitch)
                    part_lines.append(f"{duration_str}{pitch_str}")
                    
                elif isinstance(element, note.Rest):
                    duration_str = _duration_to_kern(element.duration)
                    part_lines.append(f"{duration_str}r")
                    
                elif isinstance(element, chord.Chord):
                    duration_str = _duration_to_kern(element.duration)
                    pitches = ' '.join(_pitch_to_kern(p) for p in element.pitches)
                    part_lines.append(f"{duration_str}{pitches}")
            
            # End marker
            part_lines.append("*-")
            
            parts_data.append(part_lines)
            max_lines = max(max_lines, len(part_lines))
        
        # Pad shorter parts with empty strings
        for part_lines in parts_data:
            while len(part_lines) < max_lines:
                part_lines.append(".")
        
        # Join parts horizontally with tabs (standard **kern multi-staff format)
        kern_lines = []
        for i in range(max_lines):
            line_parts = [parts_data[j][i] for j in range(len(parts_data))]
            kern_lines.append("\t".join(line_parts))
        
        kern_output = "\n".join(kern_lines)
        
        if not kern_output or len(kern_lines) <= 2:
            raise ValueError("No musical content extracted from MusicXML")
        
        logger.info(f"Successfully converted MusicXML to **kern ({len(score.parts)} parts, {len(kern_lines)} lines)")
        return kern_output
        
    except Exception as e:
        logger.error(f"Failed to convert MusicXML to **kern: {e}")
        raise ValueError(f"MusicXML to **kern conversion failed: {e}")


def convert_musicxml_file_to_kern(musicxml_path: str, output_path: Optional[str] = None) -> str:
    """Convert MusicXML file to **kern format using verovio.
    
    Args:
        musicxml_path: Path to input MusicXML file
        output_path: Optional path to save **kern output
        
    Returns:
        **kern format string
        
    Raises:
        ValueError: If conversion fails
    """
    try:
        # Read MusicXML file
        with open(musicxml_path, 'r', encoding='utf-8') as f:
            musicxml_content = f.read()
        
        # Convert using verovio
        kern_output = convert_musicxml_to_kern(musicxml_content)
        
        # Optionally save to file
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(kern_output)
            logger.info(f"Saved **kern output to: {output_path}")
        
        return kern_output
        
    except Exception as e:
        logger.error(f"Failed to convert MusicXML file to **kern: {e}")
        raise ValueError(f"MusicXML to **kern conversion failed: {e}")
