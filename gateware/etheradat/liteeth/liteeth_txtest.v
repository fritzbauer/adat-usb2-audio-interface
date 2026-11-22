/* Machine-generated using Migen */
module liteeth_txtest(
	output tx_en,
	output [1:0] txd,
	input eth_tx_clk,
	input eth_tx_rst
);

wire gen_sink_valid;
wire gen_sink_ready;
reg gen_sink_first = 1'd0;
wire gen_sink_last;
wire [7:0] gen_sink_payload_data;
reg gen_sink_payload_be = 1'd0;
reg gen_sink_payload_error = 1'd0;
reg [6:0] gen_idx = 7'd0;
reg [23:0] gen_gap = 24'd0;
wire shim_sink_valid;
wire shim_sink_ready;
wire shim_sink_first;
wire shim_sink_last;
wire [7:0] shim_sink_payload_data;
wire shim_sink_payload_be;
wire shim_sink_payload_error;
reg [7:0] shim_shift = 8'd0;
reg [2:0] shim_count = 3'd0;
reg [7:0] self;

// synthesis translate_off
reg dummy_s;
initial dummy_s <= 1'd0;
// synthesis translate_on

assign shim_sink_valid = gen_sink_valid;
assign gen_sink_ready = shim_sink_ready;
assign shim_sink_first = gen_sink_first;
assign shim_sink_last = gen_sink_last;
assign shim_sink_payload_data = gen_sink_payload_data;
assign shim_sink_payload_be = gen_sink_payload_be;
assign shim_sink_payload_error = gen_sink_payload_error;
assign gen_sink_valid = (gen_gap == 1'd0);
assign gen_sink_payload_data = self;
assign gen_sink_last = (gen_idx == 7'd71);
assign txd = shim_shift[1:0];
assign tx_en = (shim_count != 1'd0);
assign shim_sink_ready = (shim_count <= 1'd1);

// synthesis translate_off
reg dummy_d;
// synthesis translate_on
always @(*) begin
	self <= 8'd0;
	case (gen_idx)
		1'd0: begin
			self <= 7'd85;
		end
		1'd1: begin
			self <= 7'd85;
		end
		2'd2: begin
			self <= 7'd85;
		end
		2'd3: begin
			self <= 7'd85;
		end
		3'd4: begin
			self <= 7'd85;
		end
		3'd5: begin
			self <= 7'd85;
		end
		3'd6: begin
			self <= 7'd85;
		end
		3'd7: begin
			self <= 8'd213;
		end
		4'd8: begin
			self <= 8'd255;
		end
		4'd9: begin
			self <= 8'd255;
		end
		4'd10: begin
			self <= 8'd255;
		end
		4'd11: begin
			self <= 8'd255;
		end
		4'd12: begin
			self <= 8'd255;
		end
		4'd13: begin
			self <= 8'd255;
		end
		4'd14: begin
			self <= 2'd2;
		end
		4'd15: begin
			self <= 8'd173;
		end
		5'd16: begin
			self <= 7'd110;
		end
		5'd17: begin
			self <= 7'd101;
		end
		5'd18: begin
			self <= 7'd116;
		end
		5'd19: begin
			self <= 1'd1;
		end
		5'd20: begin
			self <= 8'd153;
		end
		5'd21: begin
			self <= 8'd153;
		end
		5'd22: begin
			self <= 7'd84;
		end
		5'd23: begin
			self <= 7'd88;
		end
		5'd24: begin
			self <= 7'd71;
		end
		5'd25: begin
			self <= 7'd69;
		end
		5'd26: begin
			self <= 7'd78;
		end
		5'd27: begin
			self <= 7'd84;
		end
		5'd28: begin
			self <= 7'd69;
		end
		5'd29: begin
			self <= 7'd83;
		end
		5'd30: begin
			self <= 7'd84;
		end
		5'd31: begin
			self <= 1'd0;
		end
		6'd32: begin
			self <= 1'd0;
		end
		6'd33: begin
			self <= 1'd0;
		end
		6'd34: begin
			self <= 1'd0;
		end
		6'd35: begin
			self <= 1'd0;
		end
		6'd36: begin
			self <= 1'd0;
		end
		6'd37: begin
			self <= 1'd0;
		end
		6'd38: begin
			self <= 1'd0;
		end
		6'd39: begin
			self <= 1'd0;
		end
		6'd40: begin
			self <= 1'd0;
		end
		6'd41: begin
			self <= 1'd0;
		end
		6'd42: begin
			self <= 1'd0;
		end
		6'd43: begin
			self <= 1'd0;
		end
		6'd44: begin
			self <= 1'd0;
		end
		6'd45: begin
			self <= 1'd0;
		end
		6'd46: begin
			self <= 1'd0;
		end
		6'd47: begin
			self <= 1'd0;
		end
		6'd48: begin
			self <= 1'd0;
		end
		6'd49: begin
			self <= 1'd0;
		end
		6'd50: begin
			self <= 1'd0;
		end
		6'd51: begin
			self <= 1'd0;
		end
		6'd52: begin
			self <= 1'd0;
		end
		6'd53: begin
			self <= 1'd0;
		end
		6'd54: begin
			self <= 1'd0;
		end
		6'd55: begin
			self <= 1'd0;
		end
		6'd56: begin
			self <= 1'd0;
		end
		6'd57: begin
			self <= 1'd0;
		end
		6'd58: begin
			self <= 1'd0;
		end
		6'd59: begin
			self <= 1'd0;
		end
		6'd60: begin
			self <= 1'd0;
		end
		6'd61: begin
			self <= 1'd0;
		end
		6'd62: begin
			self <= 1'd0;
		end
		6'd63: begin
			self <= 1'd0;
		end
		7'd64: begin
			self <= 1'd0;
		end
		7'd65: begin
			self <= 1'd0;
		end
		7'd66: begin
			self <= 1'd0;
		end
		7'd67: begin
			self <= 1'd0;
		end
		7'd68: begin
			self <= 7'd81;
		end
		7'd69: begin
			self <= 6'd37;
		end
		7'd70: begin
			self <= 7'd83;
		end
		default: begin
			self <= 8'd209;
		end
	endcase
// synthesis translate_off
	dummy_d <= dummy_s;
// synthesis translate_on
end

always @(posedge eth_tx_clk) begin
	if ((gen_sink_valid & gen_sink_ready)) begin
		if ((gen_idx == 7'd71)) begin
			gen_idx <= 1'd0;
			gen_gap <= 23'd4999928;
		end else begin
			gen_idx <= (gen_idx + 1'd1);
		end
	end else begin
		if ((gen_gap != 1'd0)) begin
			gen_gap <= (gen_gap - 1'd1);
		end
	end
	if ((shim_sink_valid & shim_sink_ready)) begin
		shim_shift <= shim_sink_payload_data;
		shim_count <= 3'd4;
	end else begin
		if ((shim_count != 1'd0)) begin
			shim_shift <= (shim_shift >>> 2'd2);
			shim_count <= (shim_count - 1'd1);
		end
	end
	if (eth_tx_rst) begin
		gen_idx <= 7'd0;
		gen_gap <= 24'd0;
		shim_shift <= 8'd0;
		shim_count <= 3'd0;
	end
end

endmodule
