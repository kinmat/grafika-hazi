//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
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
// Nev    : Matok Kinga
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
enum MaterialType { ROUGH, REFLECTIVE };

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) + (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
	mat4 Q; // symmetric matrix
	float f(vec4 r) { // r.w = 1
		return dot(r * Q, r);
	}
	vec3 gradf(vec4 r) { // r.w = 1
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}

public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Hiperboloid : public Intersectable {
	vec3 center;
	float d;
	float e;
	float f;
	float hossz;

	Hiperboloid(const vec3& _center, vec4 abc, Material* _material) {
		center = _center;
		d = abc.x;
		e = abc.y;
		f = abc.z;
		hossz = abc.w;
		material = _material;
		Q = mat4(1 / (d * d), 0, 0, 0,
			0, -1 / (e * e), 0, 0,
			0, 0, 1 / (f * f), 0,
			0, 0, 0, 1);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		float a = ray.dir.x * ray.dir.x / (d * d) - ray.dir.y * ray.dir.y / (e * e) + ray.dir.z * ray.dir.z / (f * f);
		float b = 2.0f * ray.dir.x * dist.x / (d * d) - 2.0f * ray.dir.y * dist.y / (e * e) + 2.0f * ray.dir.z * dist.z / (f * f);
		float c = dist.x * dist.x / (d * d) - dist.y * dist.y / (e * e) + dist.z * dist.z / (f * f) - 1;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		if (t2 > 0) {
			hit.t = t2;
		}
		else hit.t = t1;
		//	hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		vec4 hitpos(hit.position.x, hit.position.y, hit.position.z, 1);
		vec3 u = gradf(hitpos - vec4(center.x, center.y, center.z, 1));
		hit.normal = normalize(u);
		hit.material = material;
		if (abs(hit.position.y - center.y) > hossz) hit.t = -1;
		if (hit.position.y < center.y) hit.t = -1;
		if (hit.t == -1) {
			vec3 poz2 = ray.start + ray.dir * t1;
			if (poz2.y > center.y&& abs(poz2.y - center.y) < hossz) {
				hit.t = t1;
				hit.normal = -1 * hit.normal;
			}
		}

		return hit;
	}

};

struct Hiperboloid2 : public Intersectable {
	vec3 center;
	float d;
	float e;
	float f;
	float hossz = 0.6f;

	Hiperboloid2(const vec3& _center, vec4 abc, Material* _material) {
		center = _center;
		d = abc.x;
		e = abc.y;
		f = abc.z;
		hossz = abc.w;
		material = _material;
		Q = mat4(1 / (d * d), 0, 0, 0,
			0, -1 / (e * e), 0, 0,
			0, 0, 1 / (f * f), 0,
			0, 0, 0, 1);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		float a = ray.dir.x * ray.dir.x / (d * d) - ray.dir.y * ray.dir.y / (e * e) + ray.dir.z * ray.dir.z / (f * f);
		float b = 2.0f * ray.dir.x * dist.x / (d * d) - 2.0f * ray.dir.y * dist.y / (e * e) + 2.0f * ray.dir.z * dist.z / (f * f);
		float c = dist.x * dist.x / (d * d) - dist.y * dist.y / (e * e) + dist.z * dist.z / (f * f) - 1;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		if (t2 > 0) {
			hit.t = t2;
		}
		else hit.t = t1;

		hit.position = ray.start + ray.dir * hit.t;
		vec4 hitpos(hit.position.x, hit.position.y, hit.position.z, 1);
		vec3 u = gradf(hitpos - vec4(center.x, center.y, center.z, 1));
		hit.normal = normalize(u);
		hit.material = material;
		if (abs(hit.position.y - center.y) > hossz) hit.t = -1;
		if (hit.t == -1) {
			vec3 poz2 = ray.start + ray.dir * t1;
			if (abs(poz2.y - center.y) < hossz) {
				hit.t = t1;
				hit.normal = -1 * hit.normal;
			}
		}

		return hit;
	}

};

struct Henger : public Intersectable {
	vec3 center;
	float d;
	float f;
	float hossz;

	Henger(const vec3& _center, vec3 abc, Material* _material) {
		center = _center;
		d = abc.x;
		f = abc.y;
		hossz = abc.z;
		material = _material;
		Q = mat4(1 / (d * d), 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 1 / (f * f), 0,
			0, 0, 0, 1);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		float a = ray.dir.x * ray.dir.x / (d * d) + ray.dir.z * ray.dir.z / (f * f);
		float b = 2.0f * ray.dir.x * dist.x / (d * d) + 2.0f * ray.dir.z * dist.z / (f * f);
		float c = dist.x * dist.x / (d * d) + dist.z * dist.z / (f * f) - 1;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		vec4 hitpos(hit.position.x, hit.position.y, hit.position.z, 1);
		vec3 u = gradf(hitpos - vec4(center.x, center.y, center.z, 1));
		hit.normal = normalize(u);
		hit.material = material;
		if (abs(hit.position.y - center.y) > hossz) hit.t = -1;
		if (hit.t == -1) {
			vec3 poz2 = ray.start + ray.dir * t1;
			if (abs(poz2.y - center.y) < hossz) {
				hit.t = t1;
				hit.normal = -1 * hit.normal;
			}
		}
		return hit;
	}
};

struct Ellipsoid : public Intersectable {
	vec3 center;
	float d;
	float e;
	float f;

	Ellipsoid(const vec3& _center, vec3 abc, Material* _material) {
		center = _center;
		d = abc.x;
		e = abc.y;
		f = abc.z;
		material = _material;
		Q = mat4(1 / (d * d), 0, 0, 0,
			0, 1 / (e * e), 0, 0,
			0, 0, 1 / (f * f), 0,
			0, 0, 0, 1);


	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		float a = ray.dir.x * ray.dir.x / (d * d) + ray.dir.y * ray.dir.y / (e * e) + ray.dir.z * ray.dir.z / (f * f);
		float b = 2.0f * ray.dir.x * dist.x / (d * d) + 2.0f * ray.dir.y * dist.y / (e * e) + 2.0f * ray.dir.z * dist.z / (f * f);
		float c = dist.x * dist.x / (d * d) + dist.y * dist.y / (e * e) + dist.z * dist.z / (f * f) - 1;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec4 hitpos(hit.position.x, hit.position.y, hit.position.z, 1);
		vec3 u = gradf(hitpos - vec4(center.x, center.y, center.z, 1));
		hit.normal = normalize(u);
		hit.material = material;
		if (hit.position.y > 0.95f) hit.t = -1;

		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

struct Circle {
public:
	float r;
	float A;
	std::vector<vec3> pontok;
	float rnd() { return (float)rand() / RAND_MAX; }
	void setPontok() {
		float y = 0.95;
		for (int i = 0; i < 50;i++) {
			vec3 uj(rnd() - r, y, rnd() - r);
			if ((uj.x * uj.x + uj.z + uj.z) <= r) {
				pontok.push_back(uj);
			}

		}
	}

	void calcA() {
		A = r * r * M_PI;
	}

};

Circle circle;
const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 sky;
	vec3 La;
	Light* sun;
public:
	void build() {
		vec3 eye = vec3(0, 0, 1.8f), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 90 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		La = vec3(0.1f, 0.1f, 0.1f);
		sky = vec3(0.1f, 0.4f, 0.6f);
		vec3 lightDirection(0, 1, 0.2f), Le(15, 15, 15);
		sun = new Light(lightDirection, Le);


		Material* kek = new RoughMaterial(vec3(0.3f, 0.4f, 0.4f), vec3(3, 4, 5), 80);
		Material* piros = new RoughMaterial(vec3(0.8f, 0.29f, 0.3f), vec3(3, 4, 5), 60);
		Material* para = new RoughMaterial(vec3(0.6f, 0.2f, 0.3f), vec3(3, 4, 5), 10);
		vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1, 2.7, 1.9);
		Material* arany = new ReflectiveMaterial(n, kappa);
		Material* ezust = new ReflectiveMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
		objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec3(2, 1, 2), kek));
		circle.r = sqrt(2 * 2 * (1 - (0.95f * 0.95f / (1 * 1))));
		circle.setPontok();
		circle.calcA();
		objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec3(0.3f, 0.6f, 0.3f), arany));
		objects.push_back(new Hiperboloid(vec3(0, 0.95f, 0), vec4(circle.r,0.45f, circle.r, 3.5f), ezust)); //fenycso
		objects.push_back(new Hiperboloid2(vec3(0.9, -0.4, -0.2), vec4(0.15f, 0.3f, 0.15f, 0.6f), para));
		objects.push_back(new Ellipsoid(vec3(-0.4f, -0.7f, 0.6f), vec3(0.3f, 0.15f, 0.15f), piros));
		objects.push_back(new Henger(vec3(-0.9f, -0.4, -0.5f), vec3(0.3f, 0.3f, 0.8f), ezust));


	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}

		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0 && object->intersect(ray).position.y <= 0.95) return true;
		return false;
	}

	float pontTav(vec3 egy, vec3 ketto) {
		return sqrt(pow((egy.x - ketto.x), 2) + pow((egy.y - ketto.y), 2) + pow((egy.z - ketto.z), 2));
	}
	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return sky;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return sky + sun->Le * pow(dot(ray.dir, sun->direction), 10);
		vec3 outRadiance(0, 0, 0);
		if (hit.material->type == ROUGH) {
			outRadiance = La* hit.material->ka;
			for (int i = 0; i < circle.pontok.size(); i++) {
				vec3 dir = normalize(circle.pontok[i] - hit.position);
				Ray shadowRay(hit.position + hit.normal * epsilon, dir);
				float cosTheta = dot(hit.normal, dir);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					float cosAlph = dot(vec3(0, 1, 0), dir);
					vec3 Le = trace(Ray(hit.position + hit.normal * epsilon, dir), depth + 1);
					float deltaomega = circle.A / (float)circle.pontok.size() * cosAlph / pow(pontTav(hit.position, circle.pontok[i]), 2);
					outRadiance = outRadiance + Le * hit.material->kd*deltaomega*cosTheta;
					vec3 halfway = normalize(-ray.dir + dir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shininess) * deltaomega;
				}
			}
		}
		if (hit.material->type == REFLECTIVE) {

			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}


		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;
 
	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;
 
	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;
 
	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
 
	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		

		unsigned int vbo;		
		glGenBuffers(1, &vbo);	

		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	  
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);


	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}


void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();							
}


void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}


void onMouseMotion(int pX, int pY) {
}

void onIdle() {
}