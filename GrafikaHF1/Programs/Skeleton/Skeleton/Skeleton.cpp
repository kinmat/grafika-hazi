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
// Nev    : Matók Kinga
// Neptun : 
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
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 fragmentColor;		// computed color of the current pixel

	void main() {
		fragmentColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram;
const int korcsucs = 25;
const int nTesselatedVertices = 100;

class Kor {
	unsigned int vao;
	vec2 vertices[korcsucs];
public:
	void create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
		vec2 vertices[korcsucs];
		for (int i = 0; i < korcsucs; i++) {
			float fi = (i * 2) * M_PI / korcsucs;
			vertices[i] = vec2(cosf(fi), sinf(fi));
		}

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * korcsucs,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	void draw() {
		// Set color to grey
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 0.5f, 0.5f, 0.5f); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, korcsucs /*# Elements*/);
	}

};

class Curve {
	unsigned int vaoVectorizedCurve, vboVectorizedCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;

protected:
	std::vector<vec2> csucsok;
public:
	void create() {
		//Curve
		glGenVertexArrays(1, &vaoVectorizedCurve);
		glBindVertexArray(vaoVectorizedCurve);
		glGenBuffers(1, &vboVectorizedCurve);
		glBindBuffer(GL_ARRAY_BUFFER, vboVectorizedCurve);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);

		//Control Points
		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);
		glGenBuffers(1, &vboCtrlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
	}

	~Curve() {
		glDeleteBuffers(1, &vboCtrlPoints); glDeleteVertexArrays(1, &vaoCtrlPoints);
		glDeleteBuffers(1, &vboVectorizedCurve);glDeleteVertexArrays(1, &vaoVectorizedCurve);
	}

	void AddControlPoint(float cX, float cY) {
		if (csucsok.size() < 3)
			csucsok.push_back(vec2(cX, cY));
	}
	mat4 M() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	vec2 sideC(int p1, int p2) {
		float a = 1 + csucsok[p1].x * csucsok[p1].x + csucsok[p1].y * csucsok[p1].y;
		float b = 1 + csucsok[p2].x * csucsok[p2].x + csucsok[p2].y * csucsok[p2].y;
		float d = csucsok[p1].x * 2;
		float e = csucsok[p1].y * 2;
		float f = csucsok[p2].x * 2;
		float g = csucsok[p2].y * 2;

		float cY = (b * d - f * a) / (g * d - f * e);
		float cX = (a - e * cY) / d;
		vec2 c = vec2(cX, cY);
		return c;
	}

	float sideR(vec2 c) {
		return sqrt(pow(c.x, 2) + pow(c.y, 2) - 1);
	}

	float pontTav(vec2 egy, vec2 ketto) {
		return sqrt(pow((egy.x - ketto.x), 2) + pow((egy.y - ketto.y), 2));
	}

	vec2 getAtanData(int p1, int p2) {
		float eltolas;
		vec2 c = sideC(p1, p2);
		float pontokkozt2 = pow(pontTav(csucsok[p1], csucsok[p2]), 2);
		float r = sideR(c);
		float elhossz = acosf(1 - pontokkozt2 / (2 * r * r));

		vec2 nul = vec2(c.x + r, c.y);

		float nulP1 = pontTav(nul, csucsok[p1]);
		float nulP1szog = acosf(1 - pow(nulP1, 2) / (2 * r * r));


		float nulP2 = pontTav(nul, csucsok[p2]);
		float nulP2szog = acosf(1 - pow(nulP2, 2) / (2 * r * r));

		if (csucsok[p1].y > c.y&& csucsok[p2].y > c.y) {
			if (csucsok[p1].x > csucsok[p2].x) eltolas = nulP1szog;
			else eltolas = nulP2szog;
		}
		else if (csucsok[p1].y < c.y && csucsok[p2].y < c.y) {
			if (csucsok[p1].x < csucsok[p2].x) eltolas = -nulP1szog + 2 * M_PI;
			else eltolas = -nulP2szog + 2 * M_PI;
		}
		else if (csucsok[p1].y < c.y) eltolas = -nulP1szog + 2 * M_PI;
		else eltolas = -nulP2szog + 2 * M_PI;
		return vec2(elhossz, eltolas);
	}

	float korSzog(vec2 c1, vec2 c2) {
		float r1 = sideR(c1);
		float r2 = sideR(c2);
		float d = pontTav(c1, c2);
		float kosz = fabs(r1 * r1 + r2 * r2 - d * d) / (2 * r1 * r2);
		return acosf(kosz) * 180 / M_PI;
	}

	float sziriuszKerulet(vec2 c1, vec2 c2) {
		float x1 = c1.x;
		float y1 = c1.y;
		float dx = c2.x - x1;
		float dy = c2.y - y1;
		float szaml = sqrt(dx * dx + dy * dy);
		float nev = 1 - x1 * x1 - y1 * y1;
		return fabs(szaml / nev);
	}

	void Draw() {
		mat4 MVPTransform = M();
		gpuProgram.setUniform(MVPTransform, "MVP");

		if (csucsok.size() > 0) {
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, csucsok.size() * sizeof(vec2), &csucsok[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(1, 0, 0), "color");
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, csucsok.size());
		}

		if (csucsok.size() == 3) {
			std::vector<vec2> origok;
			origok.push_back(sideC(0, 2));
			origok.push_back(sideC(0, 1));
			origok.push_back(sideC(2, 1));


			std::vector<vec2> gorbeData;
			std::vector<vec2> atanData;
			atanData.push_back(getAtanData(0, 2));
			atanData.push_back(getAtanData(0, 1));
			atanData.push_back(getAtanData(2, 1));
			std::vector<vec2> gorbe;


			for (int j = 0; j < 3; j++) {
				for (int i = 0; i < nTesselatedVertices; i++) {
					float fi = i * atanData[j].x / nTesselatedVertices + atanData[j].y;
					float r = sideR(origok[j]);
					float x = origok[j].x + r * cosf(fi);
					float y = origok[j].y + r * sinf(fi);
					gorbeData.push_back(vec2(x, y));
					gorbe.push_back(vec2(x, y));
				}
			}
			
			printf("Alpha: %f ", korSzog(origok[0], origok[1]));
			printf("Beta: %f ",korSzog(origok[1], origok[2]));
			printf("Gamma: %f ", korSzog(origok[0], origok[2]));
			printf("Angle sum: %f \n", korSzog(origok[0], origok[1]) + korSzog(origok[1], origok[2]) + korSzog(origok[0], origok[2]));				printf("a : %f ", sziriuszKerulet(origok[0], origok[1]));
			printf("b : %f ", sziriuszKerulet(origok[1], origok[2]));
			printf("c : %f\n", sziriuszKerulet(origok[0], origok[2]));
			
			glBindVertexArray(vaoVectorizedCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboVectorizedCurve);
			glBufferData(GL_ARRAY_BUFFER, gorbeData.size() * sizeof(vec2), &gorbeData[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(1, 1, 0.2f), "color");
			glLineWidth(2.0f);
			glDrawArrays(GL_LINE_STRIP, 0, gorbeData.size());

		}
	}
};

Curve* curve;
Kor kor;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	kor.create();
	glLineWidth(2.0f);
	//	vonal.create();
	curve = new Curve();
	curve->create();
	//	curve = new Curve();
		// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer
	kor.draw();
	curve->Draw();
	glutSwapBuffers(); // exchange buffers for double buffering

}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		curve->AddControlPoint(cX, cY);
		glutPostRedisplay();
	}

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
