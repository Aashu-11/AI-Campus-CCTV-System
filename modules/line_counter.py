class LineCounter:
    """Line crossing counter for entry/exit tracking"""
    
    def __init__(self, line_start, line_end):
        self.line_start = line_start
        self.line_end = line_end
        self.crossed_objects = set()
        self.entry_count = 0
        self.exit_count = 0
        
    def update(self, objects_info):
        """Update line crossing counts"""
        for object_id_str, info in objects_info.items():
            object_id = int(object_id_str)
            centroid = info['centroid']
            positions = info['positions_history']
            
            if len(positions) < 2:
                continue
                
            # Check if object crossed the line
            prev_pos = positions[-2]
            curr_pos = positions[-1]
            
            if self.line_crossed(prev_pos, curr_pos):
                if object_id not in self.crossed_objects:
                    self.crossed_objects.add(object_id)
                    # Determine direction
                    if self.get_crossing_direction(prev_pos, curr_pos) > 0:
                        self.entry_count += 1
                    else:
                        self.exit_count += 1
    
    def line_crossed(self, p1, p2):
        """Check if line segment p1-p2 crosses the counting line"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = self.line_start
        x4, y4 = self.line_end
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return False
            
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def get_crossing_direction(self, p1, p2):
        """Get direction of crossing"""
        line_vec = (self.line_end[0] - self.line_start[0], self.line_end[1] - self.line_start[1])
        movement_vec = (p2[0] - p1[0], p2[1] - p1[1])
        return line_vec[0] * movement_vec[1] - line_vec[1] * movement_vec[0]