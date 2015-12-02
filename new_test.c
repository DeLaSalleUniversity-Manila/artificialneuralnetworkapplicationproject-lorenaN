/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2012 Steffen Nissen (sn@leenissen.dk)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>

#include "fann.h"

int main()
{
	fann_type *calc_out;
	unsigned int i;
	unsigned int j;
	unsigned int k;
	unsigned int max_output;
	unsigned int max_expected;
	unsigned int correct = 0;
	unsigned int wrong = 0;
	int ret = 0;

	struct fann *ann;
	struct fann_train_data *data;

	printf("Creating network.\n");

#ifdef FIXEDFANN
	ann = fann_create_from_file("new_fixed.net");
#else
	ann = fann_create_from_file("new_float.net");
#endif

	if(!ann)
	{
		printf("Error creating ann --- ABORTING.\n");
		return -1;
	}

	fann_print_connections(ann);
	fann_print_parameters(ann);

	printf("Testing network.\n");

#ifdef FIXEDFANN
	data = fann_read_train_from_file("new_fixed.data");
#else
	data = fann_read_train_from_file("new_test.data");
#endif

	for(i = 0; i < fann_length_train_data(data); i++)
	{
		fann_reset_MSE(ann);
		calc_out = fann_test(ann, data->input[i], data->output[i]);
				
		max_output = 0;
		for(j = 1; j < 4; j++)
		{
			if(calc_out[j] > calc_out[j-1])
				max_output = j;
		}
		
		max_expected = 0;
		for(k = 1; k < 4; k++)
		{
			if(data->output[i][k] > data->output[i][k-1])
				max_expected = k;
		}
		
#ifdef FIXEDFANN
		printf("\nTest: %d\nP0: %d\nP1: %d\nP2: %d\nP3: %d\nExpected no. of sections: %d\nOutput no. of sections: %d\n",
			   i, calc_out[0], calc_out[1], calc_out[2], calc_out[3], max_expected, max_output );

		if((float) fann_abs(calc_out[0] - data->output[i][0]) / fann_get_multiplier(ann) > 0.2)
		{
			printf("Test failed\n");
			ret = -1;
		}
#else
		printf("\nTest: %d\nP0: %f\nP1: %f\nP2: %f\nP3: %f\nExpected no. of sections: %d\nOutput no. of sections: %d\n",
			   i, calc_out[0], calc_out[1], calc_out[2], calc_out[3], max_expected, max_output );
		
		if(max_expected == max_output)
			correct++;
		else
			wrong++;
			   
#endif
	}
	printf("\nCorrect: %d\nWrong: %d\n", correct, wrong);
	printf("Accuracy: %f\n", (float) correct/(correct+wrong)*100);
	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);

	return ret;
}
