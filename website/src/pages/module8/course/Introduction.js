import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Image Processing</h1>
      <p>
        In this section, you will understand the fundamental concepts of image
        processing.
      </p>
      <Row>
        <Col>
          <h2>Overview of Image Processing and Its Applications</h2>
          <p>
            Image processing is a field that deals with the analysis,
            manipulation, and understanding of digital images. It has a wide
            range of applications, including medical imaging, computer vision,
            and digital photography.
          </p>
          <h2>Basics of Digital Images</h2>
          <p>
            Digital images are represented as a grid of pixels, where each pixel
            has a specific color value. The color model used determines the
            number of bits required to represent each pixel. Common color models
            include RGB, Grayscale, and HSV. Image formats such as JPEG, PNG,
            and TIFF are used to store and transmit digital images.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
