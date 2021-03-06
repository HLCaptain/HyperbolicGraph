//=============================================================================================
// Mintaprogram: Z?ld h?romsz?g. Ervenyes 2019. osztol.
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
// Nev    : P?sp?k-Kiss Bal?zs
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
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;							// copy texture coordinates
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform sampler2D textureUnit;
	uniform int isGPUProcedural;
	uniform vec3 color;		// uniform variable, the color of the primitive
	uniform vec3 randomColor1;
	uniform vec3 randomColor2;

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	
	out vec4 outColor;		// computed color of the current pixel

	vec3 ProceduralSquareColor(vec2 c) {
		if (c.x > 0.25 && c.x < 0.75 && c.y > -0.25 && c.y < 0.25) {
			return randomColor1;
		} else return randomColor2;
	}

	void main() {
		if (isGPUProcedural != 0) {
			outColor = vec4(ProceduralSquareColor(texCoord), 1);
		} else {
			outColor = vec4(color, 1);	// computed color is the color of the primitive
		}
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders

// Graph information
const int numberOfVertices = 50; // 50 vertices in the graph.
const float edgeChance = 0.05f; // 5% is the chance of an edge being in between 2 points.

// Lorents multiplication (dot product with the z multiplication being negated)
float Lorentz(const vec3 v1, const vec3 v2) { return (v1.x * v2.x + v1.y * v2.y - v1.z * v2.z); }

// HyperPoint has positions on Beltrami-Klein model and Hypebolic plane
class HyperPoint {
public:
	vec2 position; // real position
	vec2 bkproj; // projected onto the beltrami-klein disc
	std::vector<HyperPoint*> pairs; // the points which are connected to this point by an edge

	// vec2 is enough because w can be calculated with x and y
	HyperPoint(const vec2 position = vec2(0, 0)) : position(position) { bkproj = getBKProj(); }
	~HyperPoint() {	}
	// Getting the ambient coordinate, according to the hyperbolic x and y coordinates.
	float getW() const { return sqrtf(position.x * position.x + position.y * position.y + 1); }
	// Project the hyperbolic point onto the Beltrami-Klein disc from the Origo
	vec2 getBKProj() { return position / getW(); }
	// Getting the hyperbolic position as a vec3
	vec3 get3dPos() const { return vec3(position.x, position.y, getW()); }
	// Update the position as well as Beltrami-Klein projected position
	void updatePos(const vec2& newPos) { position = newPos; bkproj = getBKProj(); }
	void updatePos() { bkproj = getBKProj(); }
};

// Calculate distance on the hyperbolic plane between 2 points
float hyperDistance(const HyperPoint& p, const HyperPoint& q) { 
	return acoshf((-1) * Lorentz(p.get3dPos(), q.get3dPos())); 
}

// Get the hyperbolic projection on a point, which is on the Beltrami-Klein disc
vec3 getHyperProjFromBK(const vec3& p) {
	if (1 - p.x * p.x - p.y * p.y > 0.01f) { // boundary check is present
		return p / (sqrtf(1 - p.x * p.x - p.y * p.y));
	} else return p / sqrtf(0.01f);
}

// Calculating the direction vector from one point to another
vec3 hyperDirectionVector(const HyperPoint& p, const HyperPoint& q) {
	float distance = hyperDistance(p, q);
	return (q.get3dPos() - p.get3dPos() * coshf(distance)) / fmaxf(sinhf(distance), .001f); // boundary check is present
}

// Offseting a p point in v direction with a distance.
void hyperOffset(HyperPoint& p, const vec3& v, const float distance) {
	vec3 newPos = p.get3dPos() * coshf(distance) + v * sinhf(distance); // r(d) = p * cosh(d) + v * sinh(d)
	// updating the position according to the new point's position
	p.updatePos(vec2(newPos.x, newPos.y));
}

// Takes in 2 points which the transition should be based on. Transforms 'point'.
// DEFORMATION PRESENT, FIX NOT NEEDED, PRIORITY: NONE
void pan(const HyperPoint& p, const HyperPoint& q, HyperPoint& point) {
	// Don't need to pan when the two points are the same.
	// Don't pan if the points are nan
	if (p.position.x == q.position.x && p.position.y == q.position.y ||
		isnan(p.position.x) ||
		isnan(p.position.y) ||
		isnan(q.position.x) ||
		isnan(q.position.y)) { // return if found a nan coordinate in p or q
		return;
	}
	// Mirrored point is temporary
	HyperPoint tmp;
	tmp.updatePos(point.position);
	// Mirroring to p
	float distance = hyperDistance(tmp, q);
	vec3 dirVec = hyperDirectionVector(tmp, q);
	hyperOffset(tmp, dirVec, distance * 2); // Double the distance
	// Mirroring to q
	distance = hyperDistance(tmp, p);
	dirVec = hyperDirectionVector(tmp, p);
	hyperOffset(tmp, dirVec, distance * 2); // Double the distance
	// Halving the distance between tmp and this vertex, so we get the point we need
	distance = hyperDistance(point, tmp);
	dirVec = hyperDirectionVector(point, tmp);
	hyperOffset(point, dirVec, distance / 2); // Half the distance
}

// Default radius of the circle on the hyperbolic plane.
const float circleDefaultRadius = 0.1f;

// HyperCircle is a circle on the hyperbolic plane.
class HyperCircle {
public:
	HyperPoint center; // center of the circle
	std::vector<HyperPoint> hyperVertices; // vertices of the circle
	std::vector<vec2> bkproj; // Beltrami-Klein disc projection of the HyperPoints
	int tessellatedVertices; // number of vertices in the circle
	float radius; // radius
	unsigned int vao, vbo[2]; // vao for the circle, 2 vbo for the vertices and uvs
	std::vector<vec2> vertexUVs; // uv coordinates of the texture
	vec3 outerColor; // outer color of the procedural texture
	vec3 innerColor; // inner color of the procedural texture

	HyperCircle(const vec2& pos = vec2(0, 0), const int& tessellatedVertices = 16) :
		center(HyperPoint(vec2(0, 0))), // ini center as the 0,0,1
		hyperVertices(std::vector<HyperPoint>()),
		bkproj(std::vector<vec2>()),
		tessellatedVertices(tessellatedVertices),
		radius(circleDefaultRadius),
		vao(0),
		vertexUVs(std::vector<vec2>()) {
		vbo[0] = 0;
		vbo[1] = 0;
		iniCircle(tessellatedVertices, pos);
	}
	// Deleting buffers and arrays.
	~HyperCircle() {
		if (*vbo != 0) { glDeleteBuffers(2, vbo); }
		if (vao != 0) { glDeleteVertexArrays(1, &vao); }
	}
	// Initiating the circle's vertices and center. Pan it if needed.
	// Also initiates the UV coordinates.
	void iniCircle(int numberOfVertices, const vec2& pos = vec2(0, 0)) {
		// reseting everything except colors
		center.updatePos(vec2(0, 0));
		hyperVertices.clear();
		vertexUVs.clear();
		bkproj.clear();
		// tesselate circumference
		for (int i = 0; i < numberOfVertices; i++) {
			float rad = M_PI * 2 * (float) i / (float) numberOfVertices;
			vec3 dirVec = vec3(cosf(rad), sinf(rad), 0);
			HyperPoint tesselatedVertex(center.position);
			// always offseting in different directions with radius distance from center (0,0,1)
			hyperOffset(tesselatedVertex, normalize(dirVec), radius);
			// push back the needed data
			hyperVertices.push_back(tesselatedVertex);
			bkproj.push_back(tesselatedVertex.bkproj);
			// initializing UVs
			dirVec = dirVec / 2 + 0.5;
			vec2 uv(dirVec.x, dirVec.y);
			vertexUVs.push_back(uv);
		}
		// Pan circle to the new point. Has useful checks.
		panCircle(HyperPoint(vec2(0, 0)), HyperPoint(pos));
	}
	// updating the buffer
	void updateBuf() {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * bkproj.size(),  // # bytes
			&bkproj[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do change later
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * vertexUVs.size(),  // # bytes
			&vertexUVs[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do change later
	}

	// Pan every point based on two points
	void panCircle(const HyperPoint& p, const HyperPoint& q) {
		if (p.position.x == q.position.x && p.position.y == q.position.y) {
			return;
		}
		pan(p, q, center); // offseting center
		// offsetting the points on the circumference
		for (int i = 0; i < hyperVertices.size(); i++) {
			pan(p, q, hyperVertices[i]);
			bkproj[i] = hyperVertices[i].bkproj;
		}
		if (vao != 0) { // if able to, update the buffer
			updateBuf();
		}
	}
	// OpenGl initialization
	void create() {
		// Copied sections from your example codes. Modified some things though.
		// Here, we buffert the vertices
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active
		glGenBuffers(2, vbo);
		// binding vertices
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * bkproj.size(),  // # bytes
			&bkproj[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do change later
		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
		// buffering uvs
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * vertexUVs.size(),  // # bytes
			&vertexUVs[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do change later
		glEnableVertexAttribArray(1);  // AttribArray 0
		glVertexAttribPointer(1,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
		iniColor(); // setting up the random color of the circle
	}

	// Initializing the color of the circle with 3 random colours
	void iniColor() {
		innerColor = vec3(
			(float)rand() / RAND_MAX,
			(float)rand() / RAND_MAX,
			(float)rand() / RAND_MAX);
		outerColor = vec3(
			(float)rand() / RAND_MAX,
			(float)rand() / RAND_MAX,
			(float)rand() / RAND_MAX);
	}
	// Drawing out a circle with the vertices on the circumference.
	void draw() {
		// Update
		updateBuf();
		// Activate
		glBindVertexArray(vao);
		// Setting procedural texture
		gpuProgram.setUniform(innerColor, "randomColor1"); // procedural texture
		gpuProgram.setUniform(outerColor, "randomColor2"); // procedural texture
		gpuProgram.setUniform(1, "isGPUProcedural"); // procedural texture
		glDrawArrays(GL_TRIANGLE_FAN, 0, bkproj.size()); // Drawing out all vertices, which is the bkproj array
	}
	// getters, just because <i dont know>
	HyperPoint getCenter() { return center; }
	HyperPoint* getCenterPointer() { return &center; }
};

// Edge has 2 ends, which are points on the hyperbola.
class Edge {
public:
	// Storing the edge's ends in an array, because of OpenGL Buffer
	std::vector<HyperPoint*> ends;
	unsigned int vao, vbo;
	// Two HyperPoint pointers representing the two ends of the line.
	Edge(HyperPoint* start, HyperPoint* end) : ends(std::vector<HyperPoint*>()), vao(0), vbo(0) {
		ends.push_back(start);
		ends.push_back(end);
		// Add each other to the pair list of each other. Used in the force field calculations.
		start->pairs.push_back(end);
		end->pairs.push_back(start);
	}
	// Deleting buffers and arrays.
	~Edge() {
		if (vbo != 0) { glDeleteBuffers(1, &vbo); }
		if (vao != 0) { glDeleteVertexArrays(1, &vao); }
	}
	// OpenGl initialization
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
	// Drawing out a simple line between two projected points.
	void draw() {
		// Activate
		glBindVertexArray(vao);
		// Updating the buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Uploading points onto an array
		std::vector<vec2> points;
		points.push_back(ends[0]->bkproj);
		points.push_back(ends[1]->bkproj);
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			points.size() * sizeof(vec2),  // # bytes
			&points[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do change later
		glLineWidth(1.0f);
		gpuProgram.setUniform(0, "isGPUProcedural"); // we use colors, not procedural textures
		gpuProgram.setUniform(vec3(1.0f, 0.5f, 0.0f), "color"); // Orange
		glDrawArrays(GL_LINE_STRIP, 0, ends.size());
	}
};

const float verticalInterval = 24.0f; // starting x velocity for each point
const float horizontalInterval = 24.0f; // starting y velocity for each point
const float acceleration = 12.5f; // acceleration in unit/sec
const float phi = 12.0f; // variable which controls the percentage of velocity lost in every second
const float distance = 0.5f; // desired distance between vertices

class Graph {
public:
	std::vector<Edge> edges; // edges between vertices
	std::vector<HyperCircle> vertices; // vertices of the graph
	std::vector<vec2> velocity; // velocity for every vertex
	const float accel = acceleration; // reversed acceleration, which lowers velocity
	const float desiredDistance = distance; // desired distance between vertices
	Graph(int numberOfVertices, float edgeChance) : 
		edges(std::vector<Edge>()), 
		vertices(std::vector<HyperCircle>()),
		velocity(std::vector<vec2>()) {
		// Make the graph.
		iniVertices(numberOfVertices);
		iniEdges(edgeChance);
	}
	// Inicializing all vertices of the graph
	// All vertices are between -1;-1 and 1;1
	void iniVertices(int numberOfVertices) {
		srand(420); // Blaze it
		for (int i = 0; i < numberOfVertices; i++) {
			float x, y; // creating random velocity vectors for each Vertex
			x = (float)rand() / RAND_MAX * verticalInterval * 2 - verticalInterval;
			y = (float)rand() / RAND_MAX * horizontalInterval * 2 - horizontalInterval;
			vertices.push_back(HyperCircle());
			velocity.push_back(vec2(x, y));
		}
	}
	// Inicializing all edges of the graph between vertices
	void iniEdges(float edgeChance) {
		srand(69); // Nice
		for (int i = 0; i < vertices.size(); i++) {
			for (int j = i + 1; j < vertices.size(); j++) {
				if ((float)rand() / RAND_MAX < edgeChance) {
					edges.push_back(Edge(vertices[i].getCenterPointer(), vertices[j].getCenterPointer()));
				}
			}
		}
	}
	// Inicializing in an OpenGL environment, generating vaos and vbos
	void create() {
		for (int i = 0; i < edges.size(); i++) { edges[i].create(); }
		for (int i = 0; i < vertices.size(); i++) { vertices[i].create(); }
	}
	// Letting the components draw themselves.
	void draw() {
		for (int i = 0; i < edges.size(); i++) { edges[i].draw(); }
		for (int i = 0; i < vertices.size(); i++) { vertices[i].draw(); }
	}
	// Pan every vertex
	void pan(const HyperPoint& p, const HyperPoint& q) {
		for (int i = 0; i < vertices.size(); i++) { vertices[i].panCircle(p, q); }
	}
	// Updating the position and velocity of the vertices according to dtime
	void tick(long dtime) {
		float deltaSlow = accel * (float)dtime / 1000.0f; // the velocity the object needs to slow down with
		float percentageSlow = phi * (float)dtime / 1000.0f; // slow in percentage influenced by dtime
		for (int i = 0; i < vertices.size(); i++) {
			vec2 offset = velocity[i] * (float)dtime / 1000.0f;
			if (!(length(velocity[i]) > deltaSlow)) { // if the velocity would go negative, then set it to 0,0
				velocity[i] = vec2(0, 0);
			} else {
				if (length(offset) > 0.001f) { // offset when it counts, so the program wouldnt break
					if (hyperDistance(HyperPoint(offset), vertices[i].center) < 100.0f) {
						vertices[i].panCircle(HyperPoint(vec2(0, 0)), HyperPoint(offset));
					}
					velocity[i] = velocity[i] - velocity[i] * percentageSlow - normalize(velocity[i]) * deltaSlow;
				}
			}
		}
	}
	// Calculating the the net force for each Vertex, then assign the velocity based on it.
	void heuristicIteration() {
		// reseting velocities
		for (int i = 0; i < velocity.size(); i++) {
			velocity[i] = vec2(0, 0);
		}
		// Checking if all points are at 0,0. Needed to handle a case when all points are at 0,0.
		bool allPoint00 = true;
		for (int i = 0; i < vertices.size(); i++) {
			if (vertices[i].center.position.x != 0 || vertices[i].center.position.y != 0) {
				allPoint00 = false;
				break;
			}
		}
		// Generate random velocities when all points are at 0,0
		if (allPoint00) {
			for (int i = 0; i < vertices.size(); i++) {
				float x, y;
				x = (float)rand() / RAND_MAX * verticalInterval * 2 - verticalInterval;
				y = (float)rand() / RAND_MAX * horizontalInterval * 2 - horizontalInterval;
				velocity[i] = vec2(x, y);
			}
		} else { // else, calculate the net force for every point if posibble
			for (int i = 0; i < vertices.size(); i++) {
				for (int j = 0; j < vertices.size(); j++) {
					// checking if the two points are the same or are one of them are nan or indexes are the same
					if (i == j ||
						vertices[i].getCenter().position.x == vertices[j].getCenter().position.x ||
						vertices[i].getCenter().position.y == vertices[j].getCenter().position.y ||
						isnan(vertices[i].getCenter().position.x) ||
						isnan(vertices[i].getCenter().position.y) ||
						isnan(vertices[j].getCenter().position.x) ||
						isnan(vertices[j].getCenter().position.y)) { // value checks
						// the void is watching you as well as the points (beginning with the purple one) at 0,0. WAKE UP WAKE UP WAKE UP
						// (kind of) solved, but this comment is a reference to a meme, so I will just let it be here.
					} else {
						float distance = hyperDistance(vertices[j].getCenter(), vertices[i].getCenter()); // guaranteed not nan
						if (isConnected(vertices[i].center, vertices[j].center)) {
							// if the two points are connected
							float multiplier = (distance - desiredDistance);
							vec3 dirVec = hyperDirectionVector(vertices[i].getCenter(), vertices[j].getCenter());
							velocity[i] = velocity[i] - normalize(vec2(dirVec.x, dirVec.y)) * multiplier / 1.5f;
							dirVec = hyperDirectionVector(vertices[j].getCenter(), vertices[i].getCenter());
							velocity[j] = velocity[j] - normalize(vec2(dirVec.x, dirVec.y)) * multiplier / 1.5f;
						} else {
							// if the two points are not connected
							float multiplier = sqrtf(fminf(desiredDistance / distance, desiredDistance));
							vec3 dirVec = hyperDirectionVector(vertices[i].getCenter(), vertices[j].getCenter());
							velocity[i] = velocity[i] + normalize(vec2(dirVec.x, dirVec.y)) * multiplier / 2;
							dirVec = hyperDirectionVector(vertices[j].getCenter(), vertices[i].getCenter());
							velocity[j] = velocity[j] + normalize(vec2(dirVec.x, dirVec.y)) * multiplier / 2;
						}
					}
				}
			}
		}
		// reseting circle positions after calculating velocities
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i].iniCircle(vertices[i].tessellatedVertices);
		}
	}
	// Checks if two points are connected.
	bool isConnected(const HyperPoint& p1, const HyperPoint& p2) {
		for (int i = 0; i < p1.pairs.size(); i++) {
			if (p1.pairs[i] == &p2) { return true; }
		}
		return false;
	}
};

// The graph, which have some number of vertices and
// a chance to have an edge between them.
Graph graph(numberOfVertices, edgeChance);

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	gpuProgram.setUniform(0, "isGPUProcedural"); // procedural texture
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
	mat4 MVPtransf = { 1, 0, 0, 0,    // MVP matrix, 
					   0, 1, 0, 0,    // row-major!
					   0, 0, 1, 0,
					   0, 0, 0, 1 };
	gpuProgram.setUniform(MVPtransf, "MVP"); // Load a 4x4 row-major float matrix to the specified location
	graph.create(); // creating opengl things inside the graph
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
	if (key == ' ') {
		printf("\'%s\' was pressed!\n", &key);
		graph.heuristicIteration();
		glutPostRedisplay();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

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
	vec3 newPos = getHyperProjFromBK(vec3(cX, cY, 1));
	// Handling panning mouse actions properly.
	if (!mousePressed || oldPos.z == -1) { oldPos = newPos; }
	// given two points (from p to q, from oldPos to newPos), pan to the other
	graph.pan(HyperPoint(vec2(newPos.x, newPos.y)), HyperPoint(vec2(oldPos.x, oldPos.y)));
	oldPos = newPos;
	mousePressed = true;
	glutPostRedisplay(); // Redraw the scene
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (state == GLUT_UP) { mousePressed = false; }
}

// Time is previous segment, needed to calculate dtime.
long oldTime = 0;
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	// Time in ms
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program, first step
	long dtime = time - oldTime;
	// initialization fcks up panning (deform), so this is a needed latency constraint (though it works with 10fps< fine)
	if (dtime < 100) { graph.tick(dtime); } 
	oldTime = time; // last step
	glutPostRedisplay(); // Redraw the scene
}
