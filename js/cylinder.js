(function() {

  function addVertex(vertices, vi, x, y, z) {
    vertices[vi++] = x;
    vertices[vi++] = y;
    vertices[vi++] = z;
    return vi;
  };  

  function addTriangle(indices, i, v0, v1, v2) {
    indices[i++] = v0;
    indices[i++] = v1;
    indices[i++] = v2;
    return i;
  };

  function addQuad(indices, i, v00, v10, v01, v11) {
    indices[i] = v00;
    indices[i + 1] = indices[i + 5] = v10;
    indices[i + 2] = indices[i + 4] = v01;
    indices[i + 3] = v11;
    return i + 6;
  };  

  function createCylinderMesh(radius, angleSegment, trailVertexNum) {
    const vertexNum = (1 + angleSegment) * 2 + trailVertexNum * angleSegment;
    const indexNum = 3 * (angleSegment * 2 + (trailVertexNum - 1) * angleSegment * 2);

    const indices = new Int16Array(indexNum);
    const trailVertices = new Float32Array(vertexNum);
    const positions = new Float32Array(3 * vertexNum);
    const normals = new Float32Array(3 * vertexNum);

    const angleStep = 2.0 * Math.PI / angleSegment;

    let posCount = 0;
    let normalCount = 0;
    let trailVertexCount = 0;

    posCount = addVertex(positions, posCount, 0.0, 0.0, 0.0);
    normalCount = addVertex(normals, normalCount, 0.0, 0.0, -1.0);
    trailVertices[trailVertexCount++] = 0;
    for (let ti = 0; ti < trailVertexNum + 2; ti++) {
      for (let ai = 0; ai < angleSegment; ai++) {
        const angle = ai * angleStep + Math.PI * 0.5;
        const position = new Vector3(radius * Math.cos(angle), radius * Math.sin(angle), 0.0);
        posCount = addVertex(positions, posCount, position.x, position.y, position.z);
        if (ti === 0) {
          normalCount = addVertex(normals, normalCount, 0.0, 0.0, -1.0);
          trailVertices[trailVertexCount++] = 0;
        } else if (ti === trailVertexNum + 1) {
          normalCount = addVertex(normals, normalCount, 0.0, 0.0, 1.0);
          trailVertices[trailVertexCount++] = trailVertexNum - 1;
        } else {
          const normal = Vector3.norm(position);
          normalCount = addVertex(normals, normalCount, normal.x, normal.y, 0.0);
          trailVertices[trailVertexCount++] = ti - 1;
        }
      }
    }
    posCount = addVertex(positions, posCount, 0.0, 0.0, 0.0);
    normalCount = addVertex(normals, normalCount, 0.0, 0.0, 1.0);
    trailVertices[trailVertexCount++] = trailVertexNum - 1;

    let indexCount = 0;
    for (let ai = 0; ai < angleSegment; ai++) {
      const aj = ai !== angleSegment - 1 ? ai + 1 : 0;
      indexCount = addTriangle(indices, indexCount, 0, ai + 1, aj + 1);
    };
    let vertexOffset = angleSegment + 1;
    for (let ti = 0; ti < trailVertexNum - 1; ti++) {
      for (let ai = 0; ai < angleSegment; ai++) {
        const aj = ai !== angleSegment - 1 ? ai + 1 : 0;
        const tj = ti + 1;
        const v00 = ai + ti * angleSegment + vertexOffset;
        const v10 = aj + ti * angleSegment + vertexOffset;
        const v01 = ai + tj * angleSegment + vertexOffset;
        const v11 = aj + tj * angleSegment + vertexOffset;
        indexCount = addQuad(indices, indexCount, v00, v01, v10, v11);
      }
    }
    vertexOffset += angleSegment * trailVertexNum;
    for (let ai = 0; ai < angleSegment; ai++) {
      const aj = ai !== angleSegment - 1 ? ai + 1 : 0;
      indexCount = addTriangle(indices, indexCount, vertexNum - 1, aj + vertexOffset, ai + vertexOffset);
    }

    return {
      indices: indices,
      trailVertices: trailVertices,
      positions: positions,
      normals: normals
    };
  }

  function createTrailFbObj(gl, textureSize) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const positionTexture = createTexture(gl, textureSize.x, textureSize.y, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const velocityTexture = createTexture(gl, textureSize.x, textureSize.y, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, velocityTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, velocityTexture, 0);
    const upTexture = createTexture(gl, textureSize.x, textureSize.y, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, upTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, upTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      positionTexture: positionTexture,
      velocityTexture: velocityTexture,
      upTexture, upTexture
    };
  }

  const INITIALIZE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

#define PI 3.14159265359

layout (location = 0) out vec3 o_position;
layout (location = 1) out vec3 o_velocity;
layout (location = 2) out vec3 o_up;

uniform float u_maxRadius;

float random(float x){
  return fract(sin(x * 12.9898) * 43758.5453);
}

vec3 randomOnSphere(float v) {
  float z = random(v * 0.12 + 583.13) * 2.0 - 1.0;
  float phi = random(v * 0.49 + 213.85) * PI * 2.0;

  float b = sqrt(1.0 - z * z);

  return vec3(b * cos(phi), b * sin(phi), z);
}


vec3 randomInSphere(float v) {
  float z = random(v * 0.42 + 213.23) * 2.0 - 1.0;
  float phi = random(v * 0.19 + 313.98) * PI * 2.0;
  float r = random(v * 0.35 + 192.75);

  float a = pow(r, 1.0 / 3.0);
  float b = sqrt(1.0 - z * z);

  return vec3(a * b * cos(phi), a * b * sin(phi), a * z);
}

void main(void) {
  o_position = 10.0 * randomInSphere(gl_FragCoord.x);
  o_velocity = vec3(0.0);
  o_up = randomOnSphere(gl_FragCoord.x * 0.21 + random(3424.34));
}
`;

  const UPDATE_FRAGMENT_SHADDER_SOURCE =
`#version 300 es

precision highp float;

layout (location = 0) out vec3 o_position;
layout (location = 1) out vec3 o_velocity;
layout (location = 2) out vec3 o_up;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_upTexture;
uniform float u_time;
uniform float u_deltaTime;
uniform float u_maxSpeed;
uniform float u_maxForce;
uniform float u_boundaryRadius;
uniform float u_noiseScale;

float random(vec4 x){
  return fract(sin(dot(x,vec4(12.9898, 78.233, 39.425, 27.196))) * 43758.5453);
}

float valuenoise(vec4 x) {
  vec4 i = floor(x);
  vec4 f = fract(x);

  vec4 u = f * f * (3.0 - 2.0 * f);

  return mix(
    mix(
      mix(
          mix(random(i + vec4(0.0, 0.0, 0.0, 0.0)), random(i + vec4(1.0, 0.0, 0.0, 0.0)), u.x),
          mix(random(i + vec4(0.0, 1.0, 0.0, 0.0)), random(i + vec4(1.0, 1.0, 0.0, 0.0)), u.x),
          u.y
      ),
      mix(
          mix(random(i + vec4(0.0, 0.0, 1.0, 0.0)), random(i + vec4(1.0, 0.0, 1.0, 0.0)), u.x),
          mix(random(i + vec4(0.0, 1.0, 1.0, 0.0)), random(i + vec4(1.0, 1.0, 1.0, 0.0)), u.x),
          u.y
      ),
      u.z
    ),
    mix(
      mix(
          mix(random(i + vec4(0.0, 0.0, 0.0, 1.0)), random(i + vec4(1.0, 0.0, 0.0, 1.0)), u.x),
          mix(random(i + vec4(0.0, 1.0, 0.0, 1.0)), random(i + vec4(1.0, 1.0, 0.0, 1.0)), u.x),
          u.y
      ),
      mix(
          mix(random(i + vec4(0.0, 0.0, 1.0, 1.0)), random(i + vec4(1.0, 0.0, 1.0, 1.0)), u.x),
          mix(random(i + vec4(0.0, 1.0, 1.0, 1.0)), random(i + vec4(1.0, 1.0, 1.0, 1.0)), u.x),
          u.y
      ),
      u.z
    ),
    u.w
  );
}

float fbm(vec4 x) {
  float sum = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    sum += amp * valuenoise(x);
    amp *= 0.5;
    x *= 2.01;
  }
  return sum * 2.0 - 1.0;
}

float noiseX(vec4 x) {
  return fbm(x * 0.34 + vec4(4324.32, 7553.13, 5417.33, 1484.43));
}

float noiseY(vec4 x) {
  return fbm(x * 0.71 + vec4(1614.43, 8439.32, 4211.93, 8546.29));
}

float noiseZ(vec4 x) {
  return fbm(x * 0.54 + vec4(4342.34, 7569.34, 3812.42, 1589.54));
}

vec3 curlnoise(vec3 x, float time) {
  float e = 0.01;

  vec3 dx = vec3(e, 0.0, 0.0);
  vec3 dy = vec3(0.0, e, 0.0);
  vec3 dz = vec3(0.0, 0.0, e);

  return normalize(vec3(
    noiseZ(vec4(x + dy, time)) - noiseY(vec4(x + dz, time)),
    noiseX(vec4(x + dz, time)) - noiseZ(vec4(x + dx, time)),
    noiseY(vec4(x + dx, time)) - noiseX(vec4(x + dy, time))
  ));
}

vec3 limit(vec3 v, float max) {
  if (length(v) > max) {
    return normalize(v) * max;
  }
  return v;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);

  vec3 nextPosition, nextVelocity, nextUp;
  if (coord.y == 0) {
    vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;
    vec3 position = texelFetch(u_positionTexture, coord, 0).xyz;
    vec3 up = texelFetch(u_upTexture, coord, 0).xyz;

    vec3 acceleration = curlnoise(position * u_noiseScale, u_time * 0.2);

    acceleration = u_maxForce * mix(acceleration, -normalize(position), smoothstep(u_boundaryRadius, u_boundaryRadius * 1.05, length(position)));

    nextVelocity = limit(velocity + u_deltaTime * acceleration, u_maxSpeed);
    nextPosition = position + u_deltaTime * nextVelocity;

    vec3 front = normalize(nextVelocity);
    vec3 right = cross(front, up);
    nextUp = normalize(cross(right, front));
  } else {
    nextPosition = texelFetch(u_positionTexture, ivec2(coord.x, coord.y - 1), 0).xyz;
    nextVelocity = texelFetch(u_velocityTexture, ivec2(coord.x, coord.y - 1), 0).xyz;
    nextUp = texelFetch(u_upTexture, ivec2(coord.x, coord.y - 1), 0).xyz;
  }
  o_position = nextPosition;
  o_velocity = nextVelocity;
  o_up = nextUp;
}

`;

  const RENDER_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in vec3 i_position;
layout (location = 1) in vec3 i_normal;
layout (location = 2) in float i_trailVertex;

out vec3 v_normal;
out vec3 v_worldPos;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform sampler2D u_upTexture;
uniform mat4 u_vpMatrix;

mat4 getLookMat(vec3 front, vec3 up) {
  vec3 z = -normalize(front);
  vec3 x = cross(z, up);
  vec3 y = cross(x, z);

  return mat4(
    x.x, x.y, x.z, 0.0,
    y.x, y.y, y.z, 0.0,
    z.x, z.y, z.z, 0.0,
    0.0, 0.0, 0.0, 1.0
  );
}

void main(void) {
  vec3 instancePosition = texelFetch(u_positionTexture, ivec2(gl_InstanceID, int(i_trailVertex)), 0).xyz;
  vec3 velocity = texelFetch(u_velocityTexture, ivec2(gl_InstanceID, int(i_trailVertex)), 0).xyz;
  vec3 up = texelFetch(u_upTexture, ivec2(gl_InstanceID, int(i_trailVertex)), 0).xyz;

  mat4 lookMat = getLookMat(normalize(velocity), up);

  mat4 modelMatrix = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    instancePosition.x, instancePosition.y, instancePosition.z, 1.0
  ) * lookMat;

  mat4 mvpMatrix = u_vpMatrix * modelMatrix;
  v_normal = (lookMat * vec4(i_normal, 0.0)).xyz;
  v_worldPos = (modelMatrix * vec4(i_position, 1.0)).xyz;
  gl_Position = mvpMatrix * vec4(i_position, 1.0);
}
`;

  const RENDER_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_normal;
in vec3 v_worldPos;

out vec4 o_color;

uniform vec3 u_color;
uniform vec3 u_cameraPos;

vec3 LightDir1 = normalize(vec3(0.0, 0.5, 1.0));
vec3 LightDir2 = normalize(vec3(-0.5, -1.0, 0.0));

void main(void) {
  vec3 normal = normalize(v_normal);
  vec3 viewDir = normalize(u_cameraPos - v_worldPos);
  vec3 reflect = reflect(-viewDir, normal);
  vec3 color = vec3(0.0);
  {
    float dotNL = dot(normal, LightDir1);
    float dotRL = dot(reflect, LightDir1);
    vec3 diffuse = u_color * max(0.0, dotNL);
    vec3 spec = vec3(1.0) * pow(max(0.0, dotRL), 8.0);
    color += diffuse + spec;
  }
  {
    float dotNL = dot(normal, LightDir2);
    float dotRL = dot(reflect, LightDir2);
    vec3 diffuse = u_color * max(0.0, dotNL);
    vec3 spec = vec3(1.0) * pow(max(0.0, dotRL), 8.0);
    color += 0.5 * (diffuse + spec);
  }
  o_color = vec4(color, 1.0);
}
`;

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const parameters = {
    dynamic: {
      'max speed': 50.0,
      'max force': 30.0,
      'boundary radius': 200.0,
      'noise scale': 0.05,
      'color': [225, 0, 80],
    },
    static: {
      'trail num': 4096,
      'vertex num': 256,
      'cylinder radius': 2.0,
    },
    'reset': _ => reset()
  };

  const gui = new dat.GUI();
  const dynamicFolder = gui.addFolder('dynamic parameter');
  dynamicFolder.add(parameters.dynamic, 'max speed', 0.0, 100.0);
  dynamicFolder.add(parameters.dynamic, 'max force', 0.0, 100.0);
  dynamicFolder.add(parameters.dynamic, 'boundary radius', 50.0, 400.0);
  dynamicFolder.add(parameters.dynamic, 'noise scale', 0.0, 0.2).step(0.001);
  dynamicFolder.addColor(parameters.dynamic, 'color');
  const staticFolder = gui.addFolder('static parameter');
  staticFolder.add(parameters.static, 'trail num', 1, 16384).step(1);
  staticFolder.add(parameters.static, 'vertex num', 1, 1024).step(1);
  staticFolder.add(parameters.static, 'cylinder radius', 0.0, 10.0).step(0.01);
  gui.add(parameters, 'reset');

  const canvas = document.getElementById('canvas');
  const resizeCanvas = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);

  const initializeProgram = createProgramFromSource(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, INITIALIZE_FRAGMENT_SHADER_SOURCE);
  const updateProgram = createProgramFromSource(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, UPDATE_FRAGMENT_SHADDER_SOURCE);
  const renderProgram = createProgramFromSource(gl, RENDER_VERTEX_SHADER_SOURCE, RENDER_FRAGMENT_SHADER_SOURCE);

  const updateUniforms = getUniformLocations(gl, updateProgram,
    ['u_positionTexture', 'u_velocityTexture', 'u_upTexture', 'u_time', 'u_deltaTime', 'u_maxSpeed', 'u_maxForce', 'u_boundaryRadius', 'u_noiseScale']);
  const renderUniforms = getUniformLocations(gl, renderProgram, ['u_positionTexture', 'u_upTexture', 'u_velocityTexture', 'u_vpMatrix', 'u_color', 'u_cameraPos']);

  const fillScreenVao = createFillScreenVao(gl);
  const renderToFillScreen = () => {
    gl.bindVertexArray(fillScreenVao);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);
  };

  let requestId = null;
  const reset = () => {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const trailNum = parameters.static['trail num'];
    const vertexNum = parameters.static['vertex num'];

    const mesh = createCylinderMesh(parameters.static['cylinder radius'], 8, vertexNum);
    const meshPositionVbo = createVbo(gl, mesh.positions);
    const meshNormalVbo = createVbo(gl, mesh.normals);
    const meshTrailVertexVbo = createVbo(gl, mesh.trailVertices);
    const meshIbo = createIbo(gl, mesh.indices);

    const meshVao = gl.createVertexArray();
    gl.bindVertexArray(meshVao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, meshIbo);
    [[meshPositionVbo, 3, gl.FLOAT], [meshNormalVbo, 3, gl.FLOAT], [meshTrailVertexVbo, 1, gl.FLOAT]].forEach((v, i) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, v[0]);
      gl.enableVertexAttribArray(i);
      if (v[2] === gl.FLOAT) {
        gl.vertexAttribPointer(i, v[1], v[2], false, 0, 0);
      } else {
        gl.vertexAttribI4i(i, 0, 0, 0,0);
        gl.vertexAttribIPointer(i, v[1], v[2], 0, 0);
      }
    });
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    const trailTextureSize = new Vector2(trailNum, vertexNum);
    let trailFbObjR = createTrailFbObj(gl, trailTextureSize);
    let trailFbObjW = createTrailFbObj(gl, trailTextureSize);
    const swapTrailFbObj = () => {
      const tmp = trailFbObjR;
      trailFbObjR = trailFbObjW;
      trailFbObjW = tmp;
    };

    const initializeTrails = () => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, trailFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, trailTextureSize.x, trailTextureSize.y);
      gl.useProgram(initializeProgram);
      renderToFillScreen();
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapTrailFbObj();
    };

    const updateTrails = (elapsedTime, deltaTime) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, trailFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, trailTextureSize.x, trailTextureSize.y);
      gl.useProgram(updateProgram);
      setUniformTexture(gl, 0, trailFbObjR.positionTexture, updateUniforms['u_positionTexture']);
      setUniformTexture(gl, 1, trailFbObjR.velocityTexture, updateUniforms['u_velocityTexture']);
      setUniformTexture(gl, 2, trailFbObjR.upTexture, updateUniforms['u_upTexture']);
      gl.uniform1f(updateUniforms['u_time'], elapsedTime);
      gl.uniform1f(updateUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(updateUniforms['u_maxSpeed'], parameters.dynamic['max speed']);
      gl.uniform1f(updateUniforms['u_maxForce'], parameters.dynamic['max force']);
      gl.uniform1f(updateUniforms['u_boundaryRadius'], parameters.dynamic['boundary radius']);
      gl.uniform1f(updateUniforms['u_noiseScale'], parameters.dynamic['noise scale']);
      renderToFillScreen();
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapTrailFbObj();
    };

    const cameraPosition = new Vector3(0.0, 0.0, 500.0);
    const renderTrails = () => {
      const viewMatrix = Matrix4.inverse(Matrix4.lookAt(
        cameraPosition,
        Vector3.zero,
        new Vector3(0.0, 1.0, 0.0)
      ));
      const projectionMatrix = Matrix4.perspective(canvas.width / canvas.height, 60, 0.01, 1000.0);
      const vpMatrix = Matrix4.mul(viewMatrix, projectionMatrix);

      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.useProgram(renderProgram);
      setUniformTexture(gl, 0, trailFbObjR.positionTexture, renderUniforms['u_positionTexture']);
      setUniformTexture(gl, 1, trailFbObjR.velocityTexture, renderUniforms['u_velocityTexture']);
      setUniformTexture(gl, 2, trailFbObjR.upTexture, renderUniforms['u_upTexture']);
      gl.uniformMatrix4fv(renderUniforms['u_vpMatrix'], false, vpMatrix.elements);
      gl.uniform3fv(renderUniforms['u_color'], parameters.dynamic['color'].map(v => v / 255.0));
      gl.uniform3f(renderUniforms['u_cameraPos'], cameraPosition.x, cameraPosition.y, cameraPosition.z);
      gl.bindVertexArray(meshVao);
      gl.drawElementsInstanced(gl.TRIANGLES, mesh.indices.length, gl.UNSIGNED_SHORT, 0, trailNum);
      gl.bindVertexArray(null);
    };

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clearDepth(1.0);
    let elapsedTime = 0.0;
    let previousTime = performance.now();
    const render = () => {
      stats.update();

      const currentTime = performance.now();
      const deltaTime = Math.min(0.05, (currentTime - previousTime) * 0.001);
      elapsedTime += deltaTime;
      previousTime = currentTime;

      updateTrails(elapsedTime, deltaTime);
      renderTrails();

      requestId = requestAnimationFrame(render);
    }
    initializeTrails();
    render();
  };
  reset();
}());