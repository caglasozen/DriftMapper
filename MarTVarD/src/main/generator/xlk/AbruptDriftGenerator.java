package main.generator.xlk;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.*;
import moa.core.Example;
import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;

public class AbruptDriftGenerator extends AbstractOptionHandler implements InstanceStream {

	private static final long serialVersionUID = 2016053011L;

	public IntOption nAttributes = new IntOption("nAttributes", 'n', "Number of attributes as parents of the class", 5,
			1, 10);
	public IntOption nValuesPerAttribute = new IntOption("nValuesPerAttribute", 'v', "Number of values per attribute",
			3, 2, 5);
	public IntOption burnInNInstances = new IntOption("burnInNInstances", 'b',
			"Number of instances before the start of the drift", 100000, 1, Integer.MAX_VALUE);
	public FloatOption driftMagnitude = new FloatOption("driftMagnitude", 'm',
			"Magnitude of the drift between the starting probability and the one after the drift."
					+ " Magnitude is expressed as the Total Variation Distance [0,1]",
			0.5, 1e-5, 0.9);
	public FlagOption driftConditional = new FlagOption("ClassDrift", 'c',
			"States if the drift should apply to the conditional distribution p(y|x).");
	public FlagOption driftPriors = new FlagOption("CovariateDrift", 'p',
			"States if the drift should apply to the prior distribution p(x). ");
	public IntOption seed = new IntOption("seed", 'r', "Seed for random number generator", -1, Integer.MIN_VALUE,
			Integer.MAX_VALUE);

	protected InstancesHeader streamHeader;

	double[] pxbd; // p(x) before drift
	double[] pxad; // p(x) after drift

	int[] pygxbd; // p(y|x) before drift
	int[] pygxad; // p(y|x) after drift

	double sumpxbd;
	double sumpxad;

	int pLength; // length=mun of attr ^ mun of values

	RandomDataGenerator r;

	long nInstancesGeneratedSoFar;

	@Override
	public InstancesHeader getHeader() {
		FastVector<Attribute> attributes = new FastVector<>();
		List<String> attributeValues = new ArrayList<String>();
		for (int v = 0; v < nValuesPerAttribute.getValue(); v++) {
			attributeValues.add("v" + (v + 1));
		}
		for (int i = 0; i < nAttributes.getValue(); i++) {
			attributes.addElement(new Attribute("x" + (i + 1), attributeValues));
		}

		List<String> classValues = new ArrayList<String>();
		for (int v = 0; v < nValuesPerAttribute.getValue(); v++) {
			classValues.add("class" + (v + 1));
		}
		attributes.addElement(new Attribute("class", classValues));
		this.streamHeader = new InstancesHeader(
				new Instances(getCLICreationString(InstanceStream.class), attributes, 0));
		this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);

		return streamHeader;
	}

	@Override
	public long estimatedRemainingInstances() {
		// TODO Auto-generated method stub
		return -1;
	}

	@Override
	public boolean hasMoreInstances() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public Example<Instance> nextInstance() {

		double[] px;
		int[] pygx;
		double sumpx;

		if (this.nInstancesGeneratedSoFar < this.burnInNInstances.getValue()) {
			px = pxbd;
			pygx = pygxbd;
			sumpx = sumpxbd;
		} else {
			px = pxad;
			pygx = pygxad;
			sumpx = sumpxad;
		}

		Instance inst = new DenseInstance(streamHeader.numAttributes());
		inst.setDataset(streamHeader);

		// select p
		double rand = r.nextUniform(0.0, sumpx, true);
		int selectedPxIndex = 0;
		double sum = 0.0;
		do {
			if (selectedPxIndex == pLength) {
				System.out.println(
						"There is a error!!! sum=" + sum + "rand=" + rand + " NUM: " + nInstancesGeneratedSoFar);
				break;
			}
			sum = sum + px[selectedPxIndex];
			selectedPxIndex++;

		} while (rand > sum);
		selectedPxIndex--;

		// set attribute values
		int index[] = this.getIndex(selectedPxIndex);
		for (int i = 0; i < nAttributes.getValue(); i++) {
			inst.setValue(i, index[i]);
		}
		// label
		inst.setClassValue(pygx[selectedPxIndex]);

		nInstancesGeneratedSoFar++;
		return new InstanceExample(inst);

	}

	@Override
	public boolean isRestartable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void restart() {
		// TODO Auto-generated method stub

	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub

	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		nInstancesGeneratedSoFar = 0L;
		pLength = (int) Math.pow(this.nValuesPerAttribute.getValue(), this.nAttributes.getValue());
				
		pxbd = new double[pLength];
		pxad = new double[pLength];
		pygxbd = new int[pLength];
		pygxad = new int[pLength];
		RandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(seed.getValue());
		r = new RandomDataGenerator(rg);
		
		this.generatePxbdInGroup();
		if (driftPriors.isSet()) {
			this.generatePxadInGroup();
			System.out.println("m px:" + this.computeMagnitudePx());
		} else {
			pxad = pxbd;
		}

		//
		sumpxbd = 0;
		sumpxad = 0;
		for (int i = 0; i < this.pLength; i++) {
			//System.out.println("pxbd: " + pxbd[i] + "\t pxad: " + pxad[i] + "\t m: " + (pxad[i] - pxbd[i]));
			sumpxbd += pxbd[i];
			sumpxad += pxad[i];
		}
		
		this.generatePygxbd();
		if (driftConditional.isSet()) {
			this.generatePygxad();
			System.out.println("m for pygx: " + this.computeMagnitudePygx());
		} else {
			pygxad = pygxbd;
		}
		
	}

	private void generatePygxbd() {
		for (int i = 0; i < pLength; i++) {
			pygxbd[i] = r.nextInt(0, nValuesPerAttribute.getValue() - 1);
		}
	}

	private void generatePygxad() {
		int npygxToChange = (int) Math.round(driftMagnitude.getValue() * pLength);
		if (npygxToChange == 0.0) {
			System.out.println("Not enough drift to be noticeable in p(y|x) - unchanged");
			pygxad = pygxbd;
		} else {
			System.arraycopy(pygxbd, 0, pygxad, 0, pLength);
			int[] pygxToChange = r.nextPermutation(pLength, npygxToChange);
			for (int n : pygxToChange) {
				do {
					pygxad[n] = r.nextInt(0, nValuesPerAttribute.getValue() - 1);
				} while (pygxad[n] == pygxbd[n]);
			}
		}
	}

	// total variation distance
	private double computeMagnitudePx() {
		double m = 0.0;
		for (int i = 0; i < pLength; i++) {
			m += Math.abs(pxbd[i] - pxad[i]);
		}
		m /= 2;
		// System.out.println("magnitude of px:" + m);
		return m;
	}

	private double computeMagnitudePygx() {
		double m = 0.0;
		for (int i = 0; i < pLength; i++) {
			if (pygxbd[i] != pygxad[i]) {
				m = m + 1;
			}
		}
		m = m / pLength;
		// System.out.println("magnitude of pygx:" + m);
		return m;
	}

	private int[] getIndex(int position) {
		int[] index = new int[nAttributes.getValue()];
		int dividend = position;
		for (int i = 0; i < nAttributes.getValue(); i++) {
			index[index.length - 1 - i] = dividend % nValuesPerAttribute.getValue();
			dividend = dividend / nValuesPerAttribute.getValue();
		}
		return index;
	}
		
	double weight;
	int members;
	List<Integer> increasedValues;
	List<Integer> reducedValues;
	private void generatePxbdInGroup(){
		int[] randomIndex = r.nextPermutation(pLength, pLength);

		do{
			weight=r.nextGaussian(0.5, 0.2);
		}while(weight<=0||weight>=(1-this.driftMagnitude.getValue()));
		
		do{
			members=(int)Math.round(r.nextGaussian(((double)this.pLength)/2, ((double)this.pLength)/8));
		}while(members<=0||members>=this.pLength);
		
		increasedValues = new ArrayList<Integer>();
		reducedValues = new ArrayList<Integer>();
		for(int i=0;i<members;i++){
			increasedValues.add(randomIndex[i]);
		}
		for(int i=members;i<pLength;i++){
			reducedValues.add(randomIndex[i]);
		}
		
		double sum=0;
		for(int i=0;i<members;i++){
			pxbd[increasedValues.get(i)]=r.nextUniform(0, weight);
			sum+=pxbd[increasedValues.get(i)];
		}
		for(int i=0;i<members;i++){
			pxbd[increasedValues.get(i)]=pxbd[increasedValues.get(i)]/sum*weight;
		}
		sum=0;
		for(int i=0;i<reducedValues.size();i++){
			pxbd[reducedValues.get(i)]=r.nextUniform(0, (1-weight));
			sum+=pxbd[reducedValues.get(i)];
		}
		for(int i=0;i<reducedValues.size();i++){
			pxbd[reducedValues.get(i)]=pxbd[reducedValues.get(i)]/sum*(1-weight);
		}
		
	}
	
	private void generatePxadInGroup(){
		double sum=0;
		double[] addition=new double[members];
		for(int i=0;i<members;i++){
			addition[i]=r.nextUniform(0, this.driftMagnitude.getValue());
			sum+=addition[i];
		}
		for(int i=0;i<members;i++){
			addition[i]=addition[i]/sum*this.driftMagnitude.getValue();
		}
		for(int i=0;i<members;i++){
			pxad[increasedValues.get(i)]=pxbd[increasedValues.get(i)]+addition[i];
		}
		
		double rest=this.driftMagnitude.getValue();
		int loops=0;
		for(int i=0;i<reducedValues.size();i++){
			pxad[reducedValues.get(i)]=pxbd[reducedValues.get(i)];
		}
		do{
			loops++;
			for(int i=0;i<reducedValues.size();i++){
				double sub=r.nextUniform(0, pxad[reducedValues.get(i)]);
				pxad[reducedValues.get(i)]-=sub;
				rest-=sub;
				if(rest<=0){
					pxad[reducedValues.get(i)]-=rest;
					rest=0;
					break;
				}
			}
			if(rest==0) break;
		}while(true);
		//System.out.println("Loops: "+loops);
	}
	
}
