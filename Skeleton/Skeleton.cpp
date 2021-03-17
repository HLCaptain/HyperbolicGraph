//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Püspök-Kiss Balázs
// Neptun : BL6ADS
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

// Graph information
const int numberOfVertices = 200; // 50 vertices in the graph.
const float edgeChance = 0.005f; // 0...1 the chance of an edge being in between 2 points.

// Lorents multiplication (dot product with the z multiplication being negated)
float Lorentz(const vec3 v1, const vec3 v2) {
	return (v1.x * v2.x + v1.y * v2.y - v1.z * v2.z);
}

// Vertex has positions on Beltrami-Klein model and Hypebolic plane
class Vertex {
public:
	vec2 position; // real position
	vec2 bkproj; // projected onto the beltrami-klein disc
	unsigned int vao, vbo;

	// vec2 is enough because w can be calculated with x and y
	Vertex(const vec2 position = vec2(0, 0)) : position(position), vao(0), vbo(0) {
		bkproj = getBKProj();
	}

	void create() {
		// Copied sections from your example codes. Modified some things though.
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2),  // # bytes
			&bkproj,	      	// address
			GL_DYNAMIC_DRAW);	// we do change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	// Getting the ambient coordinate, according to the hyperbolic x and y coordinates.
	float getW() const { return sqrtf(position.x * position.x + position.y * position.y + 1); }
	// Project the hyperbolic point onto the Beltrami-Klein disc from the Origo
	vec2 getBKProj() { return position / getW(); }
	// Getting the hyperbolic position as a vec3
	vec3 get3dPos() const { return vec3(position.x, position.y, getW()); }
	// Update the position as well as Beltrami-Klein projected position
	void updatePos(const vec2& newPos) { position = newPos; bkproj = getBKProj(); }

	// Deleting buffers and arrays.
	~Vertex() {
		if (vbo != 0) { glDeleteBuffers(1, &vbo); }
		if (vao != 0) { glDeleteVertexArrays(1, &vao); }
	}

	// Get the hyperbolic projection on a point, which is on the Beltrami-Klein disc
	vec3 getHyperProjFromBK(const vec3& p) {
		return p / (sqrtf(1 - p.x * p.x - p.y * p.y));
	}

	void draw() {
		// Activate
		glBindVertexArray(vao);

		// Fake size of points to simulate distance from BK disc.
		// TODO: remove if done with textures
		glPointSize(fmaxf(16 / powf(getW(), 2), 2));

		gpuProgram.setUniform(vec3(1.0f, 1.0f, 0.0f), "color"); // Yellow
		glDrawArrays(GL_POINTS, 0, 1); // Drawing out 1 point, which is the bkproj
	}

	// Calculate distance on the hyperbolic plane between 2 points
	float hyperDistance(const Vertex& p, const Vertex& q) {
		float lorentz = Lorentz(p.get3dPos(), q.get3dPos());
		return acoshf((-1) * lorentz);
	}

	// Calculating the direction vector from one point to another
	vec3 hyperDirectionVector(const Vertex& p, const Vertex& q) {
		float distance = hyperDistance(p, q);
		vec3 v = (q.get3dPos() - p.get3dPos() * coshf(distance)) / sinhf(distance);
		return v;
	}

	// Offseting a p point in v direction with a distance.
	void hyperOffset(Vertex& p, const vec3& v, const float distance) {
		vec3 newPos = p.get3dPos() * coshf(distance) + v * sinhf(distance);
		p.updatePos(vec2(newPos.x, newPos.y));
	}

	// Takes in 2 points which the transition should be based on.
	// TODO: DEFORMATION PRESENT, FIX NEEDED, PRIORITY: MEDIUM
	void pan(const Vertex& p, const Vertex& q) {
		// Don't need to pan when the two points are the same.
		if (p.position.x == q.position.x && p.position.y == q.position.y) {
			return;
		}
		// Mirrored point is temporary
		Vertex tmp;
		tmp.updatePos(this->position);
		// Mirroring to p
		float distance = hyperDistance(tmp, q);
		vec3 dirVec = hyperDirectionVector(tmp, q);
		hyperOffset(tmp, dirVec, distance * 2); // Double the distance
		// Mirroring to q
		distance = hyperDistance(tmp, p);
		dirVec = hyperDirectionVector(tmp, p);
		hyperOffset(tmp, dirVec, distance * 2); // Double the distance
		// Halving the distance between tmp and this vertex
		distance = hyperDistance(*this, tmp);
		dirVec = hyperDirectionVector(*this, tmp);
		hyperOffset(*this, dirVec, distance / 2); // Half the distance

		// Updating the buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2),  // # bytes
			&bkproj,	      	// address
			GL_DYNAMIC_DRAW);	// we do change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}
};

// Edge has 2 ends, which are points on the hyperbola.
class Edge {
public:
	// Storing the edge's ends in an array, because of OpenGL Buffer
	std::vector<Vertex*> ends;
	unsigned int vao, vbo;

	Edge(Vertex* start, Vertex* end) : ends(std::vector<Vertex*>()), vao(0), vbo(0) {
		ends.push_back(start);
		ends.push_back(end);
	}

	void create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		std::vector<vec2> points;
		points.push_back(ends[0]->bkproj);
		points.push_back(ends[1]->bkproj);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			points.size() * sizeof(vec2),  // # bytes
			&points[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // 2 floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	// Deleting buffers and arrays.
	~Edge() {
		if (vbo != 0) { glDeleteBuffers(1, &vbo); }
		if (vao != 0) { glDeleteVertexArrays(1, &vao); }
	}

	void draw() {
		// Activate
		glBindVertexArray(vao);

		// Updating the buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Kind of a silly move, but don't blame me, I may fix this.
		std::vector<vec2> points;
		points.push_back(ends[0]->bkproj);
		points.push_back(ends[1]->bkproj);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			points.size() * sizeof(vec2),  // # bytes
			&points[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // 2 floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		glLineWidth(1.0f);
		gpuProgram.setUniform(vec3(1.0f, 0.5f, 0.0f), "color"); // Orange
		glDrawArrays(GL_LINE_STRIP, 0, ends.size());
	}
};

const int verticalInterval = 3;
const int horizontalInterval = 3;

class Graph {
public:
	std::vector<Edge> edges;
	std::vector<Vertex> vertices;
	Graph(int numberOfVertices, float edgeChance) : edges(std::vector<Edge>()), vertices(std::vector<Vertex>()) {
		// Make the graph.
		iniVertices(numberOfVertices);
		iniEdges(edgeChance);
	}
	// Inicializing all vertices of the graph
	// All vertices are between -1;-1 and 1;1
	void iniVertices(int numberOfVertices) {
		srand(420);
		for (int i = 0; i < numberOfVertices; i++) {
			float x, y;
			x = (float) rand() / RAND_MAX * verticalInterval * 2 - verticalInterval;
			y = (float) rand() / RAND_MAX * horizontalInterval * 2 - horizontalInterval;
			vertices.push_back(Vertex(vec2(x, y)));
		}
	}
	// Inicializing all edges of the graph between vertices
	void iniEdges(float edgeChance) {
		srand(69);
		for (int i = 0; i < vertices.size(); i++) {
			for (int j = i; j < vertices.size(); j++) {
				if ((float) rand() / (float) RAND_MAX < edgeChance) {
					edges.push_back(Edge(&vertices[i], &vertices[j]));
				}
			}
		}
	}
	// Inicializing in an OpenGL environment, generating vaos and vbos
	void create() {
		for (int i = 0; i < edges.size(); i++) {
			edges[i].create();
		}
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i].create();
		}
	}
	// Letting the components draw themselves.
	void draw() {
		for (int i = 0; i < edges.size(); i++) {
			edges[i].draw();
		}
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i].draw();
		}
	}
	// Pan every vertex
	void pan(const Vertex& p, const Vertex& q) {
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i].pan(p, q);
		}
	}
};

// The graph, which have some number of vertices and
// a chance to have an edge between them.
Graph graph(numberOfVertices, edgeChance);

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
	mat4 MVPtransf = { 1, 0, 0, 0,    // MVP matrix, 
					   0, 1, 0, 0,    // row-major!
					   0, 0, 1, 0,
					   0, 0, 0, 1 };
	gpuProgram.setUniform(MVPtransf, "MVP"); // Load a 4x4 row-major float matrix to the specified location
	graph.create();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	graph.draw(); // Drawing
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Storing old position of the mouse's projected hyperbolic position.
// Needed for transitioning into the new position.
vec3 oldPos(0, 0, -1);

// Needed to have a "smooth" panning. (no snapping)
bool mousePressed = false;

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	// newPos = point on the beltrami klein disc projected onto the hypebolic plane
	vec3 newPos = Vertex().getHyperProjFromBK(vec3(cX, cY, 1));

	// Handling panning mouse actions properly.
	if (!mousePressed || oldPos.z == -1) { oldPos = newPos; }

	// given two points (from p to q, from oldPos to newPos), pan to the other
	graph.pan(Vertex(vec2(newPos.x, newPos.y)), Vertex(vec2(oldPos.x, oldPos.y)));

	oldPos = newPos;
	mousePressed = true;
	glutPostRedisplay(); // Redraw the scene
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; mousePressed = false; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
