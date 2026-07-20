import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const steps = [
  "0", "5k", "10k", "15k", "20k", "25k", "30k", "35k",
  "40k", "45k", "50k", "55k", "60k", "61.446k", "62.5k",
];

const matrixDisplacement = {
  sgd: [
    0.2149475, 0.2224310, 0.2281815, 0.2325600, 0.2361560, 0.2390531,
    0.2416654, 0.2439720, 0.2461342, 0.2481221, 0.2499703, 0.2517544,
    0.2523116, 0.2526322,
  ],
  muon: [
    33.5626, 49.2201, 61.7035, 71.8668, 81.5277, 91.7229, 101.5992,
    111.5467, 121.0884, 129.5205, 137.6346, 144.9594, 146.9924, 148.5684,
  ],
};

const cka = {
  sgd: [
    1.0, 0.5997, 0.5942, 0.5942, 0.5975, 0.5967, 0.5977, 0.5988,
    0.6026, 0.5992, 0.6026, 0.6056, 0.6072, 0.6082, 0.6066,
  ],
  muon: [
    1.0, 0.5474, 0.5238, 0.4802, 0.4342, 0.4448, 0.4372, 0.4308,
    0.4233, 0.3862, 0.3945, 0.3565, 0.3680, 0.3438, 0.3439,
  ],
};

const r2 = {
  sgd: [
    0.90473, 0.99432, 0.99572, 0.99643, 0.99698, 0.99712, 0.99739,
    0.99751, 0.99759, 0.99767, 0.99768, 0.99773, 0.99782, 0.99783, 0.99777,
  ],
  muon: [
    0.90473, 0.98893, 0.99201, 0.99031, 0.99209, 0.98737, 0.98142,
    0.98256, 0.97274, 0.97905, 0.98381, 0.98339, 0.98418, 0.98432, 0.98412,
  ],
};

const groupRows = [
  ["Attention", "1.228×", "0.881", "189.967×", "0.013"],
  ["Embeddings", "1.799×", "0.604", "455.620×", "0.154"],
  ["MLP", "1.021×", "0.985", "139.702×", "0.017"],
  ["Unembedding", "1.359×", "0.214", "24.155×", "−0.100"],
  ["LayerNorm / auxiliary", "0.998×", "1.000", "0.962×", "0.982"],
];

function Caption({ children }: { children: string }) {
  return (
    <Text size="small" tone="tertiary" style={{ marginTop: 8 }}>
      {children}
    </Text>
  );
}

export default function Mess3FeatureMovement() {
  const theme = useHostTheme();

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={8}>
        <Row gap={10} align="center" wrap>
          <H1>MESS3 optimizer movement analysis</H1>
          <Pill size="sm" active>Seed 0</Pill>
          <Pill size="sm">15 checkpoints per run</Pill>
        </Row>
        <Text tone="secondary">
          Did SGD learn richer features by moving farther from initialization than Muon?
        </Text>
      </Stack>

      <Callout tone="warning" title="The proposed lazy-Muon mechanism is rejected">
        Muon moved the 2D parameters about 583× farther than SGD by the analyzed
        checkpoint, inflated their combined norm 147×, and changed block-3
        representation geometry more strongly. Its lower belief-probe R² therefore
        did not come from staying close to initialization.
      </Callout>

      <Grid columns={4} gap={14}>
        <Stat value="Exact" label="SGD and Muon initialization match" tone="success" />
        <Stat value="583×" label="Muon / SGD 2D displacement at 61,446" tone="warning" />
        <Stat value="147×" label="Muon 2D norm / initial 2D norm" tone="danger" />
        <Stat value="0.344" label="Muon block-3 CKA to initialization" tone="warning" />
      </Grid>

      <Grid columns="1.15fr 0.85fr" gap={16} align="stretch">
        <Card size="lg">
          <CardHeader>2D parameter displacement from initialization</CardHeader>
          <CardBody>
            <LineChart
              categories={steps.slice(1)}
              series={[
                {
                  name: "Large-batch SGD",
                  data: matrixDisplacement.sgd.map(Math.log10),
                  tone: "info",
                },
                {
                  name: "Large-batch Muon",
                  data: matrixDisplacement.muon.map(Math.log10),
                  tone: "warning",
                },
              ]}
              height={300}
              beginAtZero={false}
              showHoverGuide
            />
            <Caption>
              X-axis: optimizer updates · Y-axis: log10(||Wₜ−W₀||₂ / ||W₀||₂), all 2D parameters. Source: retained B2 checkpoints, seed 0.
            </Caption>
          </CardBody>
        </Card>

        <Card size="lg">
          <CardHeader>Interpretation</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text>
                At update 61,446, relative 2D displacement was <Text weight="semibold">0.252× for SGD</Text> versus <Text weight="semibold">146.992× for Muon</Text>.
              </Text>
              <Text>
                This is not merely radial scaling. Muon’s flattened 2D weights had
                cosine similarity <Text weight="semibold">0.019</Text> to initialization,
                versus <Text weight="semibold">0.970</Text> for SGD.
              </Text>
              <Text>
                Muon’s belief R² peaked near update 20k at 0.9921, then ended at
                0.9841 while its 2D displacement roughly doubled from 71.9× to
                148.6×.
              </Text>
              <Text
                weight="semibold"
                style={{
                  color: theme.accent.primary,
                  paddingTop: 4,
                }}
              >
                The stronger explanation is excessive continuing movement—likely an
                over-aggressive constant Muon learning rate without weight decay or
                schedule—not lazy feature learning.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Block-3 feature similarity to initialization</CardHeader>
          <CardBody>
            <LineChart
              categories={steps}
              series={[
                { name: "Large-batch SGD", data: cka.sgd, tone: "info" },
                { name: "Large-batch Muon", data: cka.muon, tone: "warning" },
              ]}
              height={270}
              yMin={0.3}
              yMax={1}
            />
            <Caption>
              X-axis: optimizer updates · Y-axis: centered linear CKA (0–1; scale-invariant). Block-3 residual stream on 4,096 fixed, evenly sampled length-10 contexts.
            </Caption>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Held-out affine belief-probe R²</CardHeader>
          <CardBody>
            <LineChart
              categories={steps}
              series={[
                { name: "Large-batch SGD", data: r2.sgd, tone: "info" },
                { name: "Large-batch Muon", data: r2.muon, tone: "warning" },
              ]}
              height={270}
              yMin={0.9}
              yMax={1}
              referenceLines={[{ value: 0.99, label: "0.99", tone: "neutral" }]}
            />
            <Caption>
              X-axis: optimizer updates · Y-axis: held-out affine belief-probe R² (0–1). Source: existing checkpoint probe results, seed 0.
            </Caption>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>Where the movement occurred at update 61,446</H2>
        <Text tone="secondary">
          Norm ratios show scaling; cosine similarity shows directional retention.
          Muon-eligible 2D groups are almost orthogonal to initialization.
        </Text>
        <Table
          headers={[
            "Parameter group",
            "SGD norm ratio",
            "SGD cosine",
            "Muon norm ratio",
            "Muon cosine",
          ]}
          rows={groupRows}
          columnAlign={["left", "right", "right", "right", "right"]}
          striped
        />
        <Caption>
          Norm ratio = ||Wₜ||₂ / ||W₀||₂. Cosine is between flattened group parameters at checkpoint and initialization. LayerNorm parameters are handled by auxiliary AdamW in the Muon recipe.
        </Caption>
      </Stack>

      <Card variant="borderless">
        <CardHeader>Method and limitations</CardHeader>
        <CardBody>
          <Stack gap={8}>
            <Text>
              The two step-0 model state dictionaries were compared tensor-by-tensor
              and were bit-identical, which is stronger evidence than matching run
              metadata alone. Both resolved runs used seed 0.
            </Text>
            <Text>
              Parameter movement is descriptive, not by itself a formal test of an
              NTK/lazy regime. Linear CKA adds a scale-invariant representation test,
              but a definitive mechanism study would rerun Muon with learning-rate
              decay and/or smaller learning rate while tracking feature kernels.
            </Text>
          </Stack>
        </CardBody>
      </Card>
    </Stack>
  );
}
