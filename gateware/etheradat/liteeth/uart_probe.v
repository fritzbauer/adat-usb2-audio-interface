/* Machine-generated using Migen */
module uart_probe(
	input rmii_crs_dv,
	input [1:0] rmii_rxd,
	output uart_txd,
	input sys_clk,
	input sys_rst,
	input eth_rx_clk,
	input eth_rx_rst,
	input negrx_clk,
	input negrx_rst,
	input eth_tx_clk,
	input eth_tx_rst,
	output rmii_tx_en,
	output [1:0] rmii_txd
);

wire rmiirxshim0_source_valid;
wire rmiirxshim0_source_ready;
wire rmiirxshim0_source_last;
wire [7:0] rmiirxshim0_source_payload_data;
wire rmiirxshim0_source_payload_be;
wire rmiirxshim0_crs_dv;
wire [1:0] rmiirxshim0_rx_data;
wire rmiirxshim0_crs_first;
wire rmiirxshim0_crs_last;
reg rmiirxshim0_crs_run = 1'd0;
reg rmiirxshim0_crs_dv_d = 1'd0;
reg [1:0] rmiirxshim0_d0 = 2'd0;
reg [1:0] rmiirxshim0_d1 = 2'd0;
reg rmiirxshim0_e0 = 1'd0;
reg rmiirxshim0_e1 = 1'd0;
wire rmiirxshim0_cap;
reg [1:0] rmiirxshim0_mode = 2'd0;
reg rmiirxshim0_f0_valid = 1'd0;
reg [7:0] rmiirxshim0_f0_data = 8'd0;
reg rmiirxshim0_f0_last = 1'd0;
reg rmiirxshim0_f1_valid = 1'd0;
reg [7:0] rmiirxshim0_f1_data = 8'd0;
reg rmiirxshim0_f1_last = 1'd0;
reg [7:0] rmiirxshim0_sreg = 8'd85;
wire rmiirxshim0_sfd_hit;
reg [2:0] rmiirxshim0_pre_cnt = 3'd0;
wire [7:0] rmiirxshim0_pre_byte;
reg [7:0] rmiirxshim0_shift = 8'd0;
reg [1:0] rmiirxshim0_bits = 2'd0;
wire rmiirxshim0_consume;
wire [7:0] rmiirxshim0_new_byte;
wire rmiirxshim0_pushing;
wire rmiirxshim1_source_valid;
wire rmiirxshim1_source_ready;
wire rmiirxshim1_source_last;
wire [7:0] rmiirxshim1_source_payload_data;
wire rmiirxshim1_source_payload_be;
wire rmiirxshim1_crs_dv;
wire [1:0] rmiirxshim1_rx_data;
wire rmiirxshim1_crs_first;
wire rmiirxshim1_crs_last;
reg rmiirxshim1_crs_run = 1'd0;
reg rmiirxshim1_crs_dv_d = 1'd0;
reg [1:0] rmiirxshim1_d0 = 2'd0;
reg [1:0] rmiirxshim1_d1 = 2'd0;
reg rmiirxshim1_e0 = 1'd0;
reg rmiirxshim1_e1 = 1'd0;
wire rmiirxshim1_cap;
reg [1:0] rmiirxshim1_mode = 2'd0;
reg rmiirxshim1_f0_valid = 1'd0;
reg [7:0] rmiirxshim1_f0_data = 8'd0;
reg rmiirxshim1_f0_last = 1'd0;
reg rmiirxshim1_f1_valid = 1'd0;
reg [7:0] rmiirxshim1_f1_data = 8'd0;
reg rmiirxshim1_f1_last = 1'd0;
reg [7:0] rmiirxshim1_sreg = 8'd85;
wire rmiirxshim1_sfd_hit;
reg [2:0] rmiirxshim1_pre_cnt = 3'd0;
wire [7:0] rmiirxshim1_pre_byte;
reg [7:0] rmiirxshim1_shift = 8'd0;
reg [1:0] rmiirxshim1_bits = 2'd0;
wire rmiirxshim1_consume;
wire [7:0] rmiirxshim1_new_byte;
wire rmiirxshim1_pushing;
wire slowtxgenerator_sink_valid;
wire slowtxgenerator_sink_ready;
reg slowtxgenerator_sink_first = 1'd0;
wire slowtxgenerator_sink_last;
wire [7:0] slowtxgenerator_sink_payload_data;
reg slowtxgenerator_sink_payload_be = 1'd0;
reg slowtxgenerator_sink_payload_error = 1'd0;
reg [6:0] slowtxgenerator_idx = 7'd0;
reg [23:0] slowtxgenerator_gap = 24'd0;
wire rmiitxshim_sink_valid;
wire rmiitxshim_sink_ready;
wire rmiitxshim_sink_first;
wire rmiitxshim_sink_last;
wire [7:0] rmiitxshim_sink_payload_data;
wire rmiitxshim_sink_payload_be;
wire rmiitxshim_sink_payload_error;
reg [7:0] rmiitxshim_byte = 8'd0;
reg [1:0] rmiitxshim_pair = 2'd0;
reg rmiitxshim_emitting = 1'd0;
reg [1:0] rmiitxshim_txd_r = 2'd0;
reg rmiitxshim_tx_en_r = 1'd0;
wire [6:0] cap_pos_wr_adr;
wire [7:0] cap_pos_wr_dat_r;
wire cap_pos_wr_we;
wire [7:0] cap_pos_wr_dat_w;
reg [6:0] cap_pos_wr_ptr = 7'd0;
reg cap_pos_pend = 1'd0;
reg cap_pos_clear_d = 1'd0;
reg cap_pos_clear = 1'd0;
reg [6:0] cap_pos_frame_len = 7'd0;
wire cap_pos_clear_in;
wire [6:0] cap_pos_rd_port_adr;
wire [7:0] cap_pos_rd_port_dat_r;
wire [6:0] cap_neg_wr_adr;
wire [7:0] cap_neg_wr_dat_r;
wire cap_neg_wr_we;
wire [7:0] cap_neg_wr_dat_w;
reg [6:0] cap_neg_wr_ptr = 7'd0;
reg cap_neg_pend = 1'd0;
reg cap_neg_clear_d = 1'd0;
reg cap_neg_clear = 1'd0;
reg [6:0] cap_neg_frame_len = 7'd0;
wire cap_neg_clear_in;
wire [6:0] cap_neg_rd_port_adr;
wire [7:0] cap_neg_rd_port_dat_r;
wire txdibitpacker_source_valid;
wire txdibitpacker_source_ready;
wire txdibitpacker_source_last;
wire [7:0] txdibitpacker_source_payload_data;
reg [7:0] txdibitpacker_shift = 8'd0;
reg [1:0] txdibitpacker_bits = 2'd0;
reg txdibitpacker_running = 1'd0;
reg [7:0] txdibitpacker_hold = 8'd0;
reg txdibitpacker_hold_valid = 1'd0;
reg txdibitpacker_hold_last = 1'd0;
wire txdibitpacker_consume;
wire [7:0] txdibitpacker_new_byte;
wire txdibitpacker_completing;
wire txdibitpacker_flushing;
wire [6:0] cap_tx_wr_adr;
wire [7:0] cap_tx_wr_dat_r;
wire cap_tx_wr_we;
wire [7:0] cap_tx_wr_dat_w;
reg [6:0] cap_tx_wr_ptr = 7'd0;
reg cap_tx_pend = 1'd0;
reg cap_tx_clear_d = 1'd0;
reg cap_tx_clear = 1'd0;
reg [6:0] cap_tx_frame_len = 7'd0;
wire cap_tx_clear_in;
wire [6:0] cap_tx_rd_port_adr;
wire [7:0] cap_tx_rd_port_dat_r;
wire txtap_source_valid;
wire txtap_source_ready;
wire txtap_source_last;
wire [7:0] txtap_source_payload_data;
wire [6:0] cap_tap_wr_adr;
wire [7:0] cap_tap_wr_dat_r;
wire cap_tap_wr_we;
wire [7:0] cap_tap_wr_dat_w;
reg [6:0] cap_tap_wr_ptr = 7'd0;
reg cap_tap_pend = 1'd0;
reg cap_tap_clear_d = 1'd0;
reg cap_tap_clear = 1'd0;
reg [6:0] cap_tap_frame_len = 7'd0;
wire cap_tap_clear_in;
wire [6:0] cap_tap_rd_port_adr;
wire [7:0] cap_tap_rd_port_dat_r;
wire uart_txd1;
wire [7:0] uarttx_data;
wire uarttx_strobe;
wire uarttx_busy;
reg [1:0] uarttx_state = 2'd0;
reg [9:0] uarttx_cnt = 10'd0;
reg [7:0] uarttx_shreg = 8'd0;
reg [2:0] uarttx_bitidx = 3'd0;
reg uarttx_pend_bit = 1'd0;
reg [7:0] uarttx_pend_data = 8'd0;
reg uarttx = 1'd0;
reg pend_pos_d = 1'd0;
reg pend_neg_d = 1'd0;
reg pend_pos = 1'd0;
reg pend_neg = 1'd0;
reg [6:0] len_pos = 7'd0;
reg [6:0] len_neg = 7'd0;
reg [25:0] hb = 26'd0;
wire hb_fire;
reg [2:0] state = 3'd0;
reg [7:0] marker = 8'd0;
reg [6:0] length = 7'd0;
reg [6:0] idx = 7'd0;
reg clear_pos = 1'd0;
reg clear_neg = 1'd0;
reg clear_tx = 1'd0;
reg clear_tap = 1'd0;
reg pend_tx_d = 1'd0;
reg pend_tx = 1'd0;
reg [6:0] len_tx = 7'd0;
reg pend_tap_d = 1'd0;
reg pend_tap = 1'd0;
reg [6:0] len_tap = 7'd0;
reg [7:0] self;

// synthesis translate_off
reg dummy_s;
initial dummy_s <= 1'd0;
// synthesis translate_on

assign rmiitxshim_sink_valid = slowtxgenerator_sink_valid;
assign slowtxgenerator_sink_ready = rmiitxshim_sink_ready;
assign rmiitxshim_sink_first = slowtxgenerator_sink_first;
assign rmiitxshim_sink_last = slowtxgenerator_sink_last;
assign rmiitxshim_sink_payload_data = slowtxgenerator_sink_payload_data;
assign rmiitxshim_sink_payload_be = slowtxgenerator_sink_payload_be;
assign rmiitxshim_sink_payload_error = slowtxgenerator_sink_payload_error;
assign uart_txd = uart_txd1;
assign hb_fire = (hb == 1'd0);
assign cap_pos_clear_in = clear_pos;
assign cap_neg_clear_in = clear_neg;
assign cap_tx_clear_in = clear_tx;
assign cap_tap_clear_in = clear_tap;
assign uarttx_strobe = ((state != 1'd0) | ((state == 1'd0) & ((((pend_pos | pend_neg) | pend_tap) | pend_tx) | hb_fire)));
assign uarttx_data = ((state == 1'd0) ? 7'd126 : ((state == 1'd1) ? 8'd170 : ((state == 2'd2) ? marker : ((state == 2'd3) ? length : ((marker == 1'd1) ? cap_pos_rd_port_dat_r : ((marker == 2'd2) ? cap_neg_rd_port_dat_r : ((marker == 3'd4) ? cap_tap_rd_port_dat_r : cap_tx_rd_port_dat_r)))))));
assign cap_pos_rd_port_adr = idx;
assign cap_neg_rd_port_adr = idx;
assign cap_tx_rd_port_adr = idx;
assign cap_tap_rd_port_adr = idx;
assign rmiirxshim0_source_payload_be = 1'd1;
assign rmiirxshim0_crs_dv = rmii_crs_dv;
assign rmiirxshim0_rx_data = rmii_rxd;
assign rmiirxshim0_crs_first = (rmiirxshim0_crs_dv & (rmiirxshim0_rx_data != 1'd0));
assign rmiirxshim0_crs_last = ((~rmiirxshim0_crs_dv) & (~rmiirxshim0_crs_dv_d));
assign rmiirxshim0_cap = (rmiirxshim0_crs_first | rmiirxshim0_crs_run);
assign rmiirxshim0_sfd_hit = (((rmiirxshim0_e1 & (rmiirxshim0_mode == 1'd0)) & (rmiirxshim0_sreg[5:0] == 5'd21)) & (rmiirxshim0_d1 == 2'd3));
assign rmiirxshim0_pre_byte = ((rmiirxshim0_pre_cnt == 3'd7) ? 8'd213 : 7'd85);
assign rmiirxshim0_source_valid = ((rmiirxshim0_mode == 1'd1) | ((rmiirxshim0_mode == 2'd2) & rmiirxshim0_f0_valid));
assign rmiirxshim0_source_payload_data = ((rmiirxshim0_mode == 1'd1) ? rmiirxshim0_pre_byte : rmiirxshim0_f0_data);
assign rmiirxshim0_source_last = ((rmiirxshim0_mode == 2'd2) & rmiirxshim0_f0_last);
assign rmiirxshim0_consume = (rmiirxshim0_source_valid & rmiirxshim0_source_ready);
assign rmiirxshim0_new_byte = {rmiirxshim0_d1, rmiirxshim0_shift[7:2]};
assign rmiirxshim0_pushing = (((rmiirxshim0_mode != 1'd0) & rmiirxshim0_e1) & (rmiirxshim0_bits == 2'd3));
assign rmiirxshim1_source_payload_be = 1'd1;
assign rmiirxshim1_crs_dv = rmii_crs_dv;
assign rmiirxshim1_rx_data = rmii_rxd;
assign rmiirxshim1_crs_first = (rmiirxshim1_crs_dv & (rmiirxshim1_rx_data != 1'd0));
assign rmiirxshim1_crs_last = ((~rmiirxshim1_crs_dv) & (~rmiirxshim1_crs_dv_d));
assign rmiirxshim1_cap = (rmiirxshim1_crs_first | rmiirxshim1_crs_run);
assign rmiirxshim1_sfd_hit = (((rmiirxshim1_e1 & (rmiirxshim1_mode == 1'd0)) & (rmiirxshim1_sreg[5:0] == 5'd21)) & (rmiirxshim1_d1 == 2'd3));
assign rmiirxshim1_pre_byte = ((rmiirxshim1_pre_cnt == 3'd7) ? 8'd213 : 7'd85);
assign rmiirxshim1_source_valid = ((rmiirxshim1_mode == 1'd1) | ((rmiirxshim1_mode == 2'd2) & rmiirxshim1_f0_valid));
assign rmiirxshim1_source_payload_data = ((rmiirxshim1_mode == 1'd1) ? rmiirxshim1_pre_byte : rmiirxshim1_f0_data);
assign rmiirxshim1_source_last = ((rmiirxshim1_mode == 2'd2) & rmiirxshim1_f0_last);
assign rmiirxshim1_consume = (rmiirxshim1_source_valid & rmiirxshim1_source_ready);
assign rmiirxshim1_new_byte = {rmiirxshim1_d1, rmiirxshim1_shift[7:2]};
assign rmiirxshim1_pushing = (((rmiirxshim1_mode != 1'd0) & rmiirxshim1_e1) & (rmiirxshim1_bits == 2'd3));
assign slowtxgenerator_sink_valid = (slowtxgenerator_gap == 1'd0);
assign slowtxgenerator_sink_payload_data = self;
assign slowtxgenerator_sink_last = (slowtxgenerator_idx == 7'd71);
assign rmii_txd = rmiitxshim_txd_r;
assign rmii_tx_en = rmiitxshim_tx_en_r;
assign rmiitxshim_sink_ready = ((~rmiitxshim_emitting) | (rmiitxshim_pair == 2'd3));
assign rmiirxshim0_source_ready = (~cap_pos_pend);
assign cap_pos_wr_adr = cap_pos_wr_ptr;
assign cap_pos_wr_dat_w = rmiirxshim0_source_payload_data;
assign cap_pos_wr_we = (rmiirxshim0_source_valid & (~cap_pos_pend));
assign rmiirxshim1_source_ready = (~cap_neg_pend);
assign cap_neg_wr_adr = cap_neg_wr_ptr;
assign cap_neg_wr_dat_w = rmiirxshim1_source_payload_data;
assign cap_neg_wr_we = (rmiirxshim1_source_valid & (~cap_neg_pend));
assign txdibitpacker_consume = (txdibitpacker_source_valid & txdibitpacker_source_ready);
assign txdibitpacker_source_valid = txdibitpacker_hold_valid;
assign txdibitpacker_source_payload_data = txdibitpacker_hold;
assign txdibitpacker_source_last = txdibitpacker_hold_last;
assign txdibitpacker_new_byte = {rmii_txd, txdibitpacker_shift[7:2]};
assign txdibitpacker_completing = (rmii_tx_en & (txdibitpacker_bits == 2'd3));
assign txdibitpacker_flushing = ((~rmii_tx_en) & txdibitpacker_running);
assign txdibitpacker_source_ready = (~cap_tx_pend);
assign cap_tx_wr_adr = cap_tx_wr_ptr;
assign cap_tx_wr_dat_w = txdibitpacker_source_payload_data;
assign cap_tx_wr_we = (txdibitpacker_source_valid & (~cap_tx_pend));
assign txtap_source_valid = (rmiitxshim_sink_valid & rmiitxshim_sink_ready);
assign txtap_source_payload_data = rmiitxshim_sink_payload_data;
assign txtap_source_last = rmiitxshim_sink_last;
assign txtap_source_ready = (~cap_tap_pend);
assign cap_tap_wr_adr = cap_tap_wr_ptr;
assign cap_tap_wr_dat_w = txtap_source_payload_data;
assign cap_tap_wr_we = (txtap_source_valid & (~cap_tap_pend));
assign uarttx_busy = (uarttx_state != 1'd0);
assign uart_txd1 = ((uarttx_state == 1'd0) ? 1'd1 : ((uarttx_state == 1'd1) ? 1'd0 : ((uarttx_state == 2'd2) ? uarttx_shreg[0] : 1'd1)));

// synthesis translate_off
reg dummy_d;
// synthesis translate_on
always @(*) begin
	self <= 8'd0;
	case (slowtxgenerator_idx)
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

always @(posedge eth_rx_clk) begin
	rmiirxshim0_crs_dv_d <= rmiirxshim0_crs_dv;
	if (rmiirxshim0_crs_first) begin
		rmiirxshim0_crs_run <= 1'd1;
	end
	if (rmiirxshim0_crs_last) begin
		rmiirxshim0_crs_run <= 1'd0;
	end
	rmiirxshim0_d0 <= rmiirxshim0_rx_data;
	rmiirxshim0_d1 <= rmiirxshim0_d0;
	rmiirxshim0_e0 <= rmiirxshim0_cap;
	rmiirxshim0_e1 <= rmiirxshim0_e0;
	if ((rmiirxshim0_e1 & (rmiirxshim0_mode == 1'd0))) begin
		rmiirxshim0_sreg <= {rmiirxshim0_sreg[5:0], rmiirxshim0_d1};
	end
	if ((rmiirxshim0_mode == 1'd0)) begin
		if (rmiirxshim0_sfd_hit) begin
			rmiirxshim0_mode <= 1'd1;
			rmiirxshim0_pre_cnt <= 1'd0;
			rmiirxshim0_bits <= 1'd0;
		end
	end else begin
		if ((rmiirxshim0_mode == 1'd1)) begin
			if (rmiirxshim0_consume) begin
				if ((rmiirxshim0_pre_cnt == 3'd7)) begin
					rmiirxshim0_mode <= 2'd2;
				end else begin
					rmiirxshim0_pre_cnt <= (rmiirxshim0_pre_cnt + 1'd1);
				end
			end
		end
	end
	if (((rmiirxshim0_mode == 2'd2) & rmiirxshim0_consume)) begin
		if (rmiirxshim0_pushing) begin
			rmiirxshim0_f0_valid <= 1'd1;
			rmiirxshim0_f0_data <= rmiirxshim0_f1_data;
			rmiirxshim0_f0_last <= rmiirxshim0_f1_last;
			rmiirxshim0_f1_valid <= 1'd1;
			rmiirxshim0_f1_data <= rmiirxshim0_new_byte;
			rmiirxshim0_f1_last <= rmiirxshim0_crs_last;
		end else begin
			rmiirxshim0_f0_valid <= rmiirxshim0_f1_valid;
			rmiirxshim0_f0_data <= rmiirxshim0_f1_data;
			rmiirxshim0_f0_last <= rmiirxshim0_f1_last;
			rmiirxshim0_f1_valid <= 1'd0;
		end
	end else begin
		if ((rmiirxshim0_pushing & (~(((rmiirxshim0_mode == 2'd2) & rmiirxshim0_f0_valid) & rmiirxshim0_f1_valid)))) begin
			if ((~rmiirxshim0_f0_valid)) begin
				rmiirxshim0_f0_valid <= 1'd1;
				rmiirxshim0_f0_data <= rmiirxshim0_new_byte;
				rmiirxshim0_f0_last <= rmiirxshim0_crs_last;
			end else begin
				if ((~rmiirxshim0_f1_valid)) begin
					rmiirxshim0_f1_valid <= 1'd1;
					rmiirxshim0_f1_data <= rmiirxshim0_new_byte;
					rmiirxshim0_f1_last <= rmiirxshim0_crs_last;
				end
			end
		end
	end
	if (((rmiirxshim0_mode != 1'd0) & rmiirxshim0_e1)) begin
		rmiirxshim0_shift <= rmiirxshim0_new_byte;
		if ((rmiirxshim0_bits == 2'd3)) begin
			rmiirxshim0_bits <= 1'd0;
		end else begin
			rmiirxshim0_bits <= (rmiirxshim0_bits + 1'd1);
		end
	end else begin
		if ((~rmiirxshim0_e1)) begin
			rmiirxshim0_bits <= 1'd0;
		end
	end
	if (eth_rx_rst) begin
		rmiirxshim0_crs_run <= 1'd0;
		rmiirxshim0_crs_dv_d <= 1'd0;
		rmiirxshim0_d0 <= 2'd0;
		rmiirxshim0_d1 <= 2'd0;
		rmiirxshim0_e0 <= 1'd0;
		rmiirxshim0_e1 <= 1'd0;
		rmiirxshim0_mode <= 2'd0;
		rmiirxshim0_f0_valid <= 1'd0;
		rmiirxshim0_f0_data <= 8'd0;
		rmiirxshim0_f0_last <= 1'd0;
		rmiirxshim0_f1_valid <= 1'd0;
		rmiirxshim0_f1_data <= 8'd0;
		rmiirxshim0_f1_last <= 1'd0;
		rmiirxshim0_sreg <= 8'd85;
		rmiirxshim0_pre_cnt <= 3'd0;
		rmiirxshim0_shift <= 8'd0;
		rmiirxshim0_bits <= 2'd0;
	end
end

always @(posedge eth_tx_clk) begin
	if ((slowtxgenerator_sink_valid & slowtxgenerator_sink_ready)) begin
		if ((slowtxgenerator_idx == 7'd71)) begin
			slowtxgenerator_idx <= 1'd0;
			slowtxgenerator_gap <= 23'd4999928;
		end else begin
			slowtxgenerator_idx <= (slowtxgenerator_idx + 1'd1);
		end
	end else begin
		if ((slowtxgenerator_gap != 1'd0)) begin
			slowtxgenerator_gap <= (slowtxgenerator_gap - 1'd1);
		end
	end
	rmiitxshim_txd_r <= (rmiitxshim_byte >>> (rmiitxshim_pair <<< 1'd1));
	rmiitxshim_tx_en_r <= rmiitxshim_emitting;
	if ((rmiitxshim_sink_valid & rmiitxshim_sink_ready)) begin
		rmiitxshim_byte <= rmiitxshim_sink_payload_data;
		rmiitxshim_pair <= 1'd0;
		rmiitxshim_emitting <= 1'd1;
	end else begin
		if (rmiitxshim_emitting) begin
			rmiitxshim_pair <= (rmiitxshim_pair + 1'd1);
			if ((rmiitxshim_pair == 2'd3)) begin
				rmiitxshim_emitting <= 1'd0;
			end
		end
	end
	if (rmii_tx_en) begin
		if ((~txdibitpacker_running)) begin
			txdibitpacker_running <= 1'd1;
			txdibitpacker_bits <= 1'd0;
		end
		txdibitpacker_shift <= txdibitpacker_new_byte;
		if (txdibitpacker_completing) begin
			txdibitpacker_bits <= 1'd0;
		end else begin
			txdibitpacker_bits <= (txdibitpacker_bits + 1'd1);
		end
	end else begin
		if (txdibitpacker_running) begin
			txdibitpacker_running <= 1'd0;
			txdibitpacker_bits <= 1'd0;
		end
	end
	if (txdibitpacker_completing) begin
		txdibitpacker_hold <= txdibitpacker_new_byte;
		txdibitpacker_hold_valid <= 1'd1;
		txdibitpacker_hold_last <= 1'd0;
	end else begin
		if (txdibitpacker_flushing) begin
			txdibitpacker_hold <= txdibitpacker_shift;
			txdibitpacker_hold_valid <= 1'd1;
			txdibitpacker_hold_last <= 1'd1;
		end else begin
			if (txdibitpacker_consume) begin
				txdibitpacker_hold_valid <= 1'd0;
				txdibitpacker_hold_last <= 1'd0;
			end
		end
	end
	if (eth_tx_rst) begin
		slowtxgenerator_idx <= 7'd0;
		slowtxgenerator_gap <= 24'd0;
		rmiitxshim_byte <= 8'd0;
		rmiitxshim_pair <= 2'd0;
		rmiitxshim_emitting <= 1'd0;
		rmiitxshim_txd_r <= 2'd0;
		rmiitxshim_tx_en_r <= 1'd0;
		txdibitpacker_shift <= 8'd0;
		txdibitpacker_bits <= 2'd0;
		txdibitpacker_running <= 1'd0;
		txdibitpacker_hold <= 8'd0;
		txdibitpacker_hold_valid <= 1'd0;
		txdibitpacker_hold_last <= 1'd0;
	end
end

always @(posedge negrx_clk) begin
	rmiirxshim1_crs_dv_d <= rmiirxshim1_crs_dv;
	if (rmiirxshim1_crs_first) begin
		rmiirxshim1_crs_run <= 1'd1;
	end
	if (rmiirxshim1_crs_last) begin
		rmiirxshim1_crs_run <= 1'd0;
	end
	rmiirxshim1_d0 <= rmiirxshim1_rx_data;
	rmiirxshim1_d1 <= rmiirxshim1_d0;
	rmiirxshim1_e0 <= rmiirxshim1_cap;
	rmiirxshim1_e1 <= rmiirxshim1_e0;
	if ((rmiirxshim1_e1 & (rmiirxshim1_mode == 1'd0))) begin
		rmiirxshim1_sreg <= {rmiirxshim1_sreg[5:0], rmiirxshim1_d1};
	end
	if ((rmiirxshim1_mode == 1'd0)) begin
		if (rmiirxshim1_sfd_hit) begin
			rmiirxshim1_mode <= 1'd1;
			rmiirxshim1_pre_cnt <= 1'd0;
			rmiirxshim1_bits <= 1'd0;
		end
	end else begin
		if ((rmiirxshim1_mode == 1'd1)) begin
			if (rmiirxshim1_consume) begin
				if ((rmiirxshim1_pre_cnt == 3'd7)) begin
					rmiirxshim1_mode <= 2'd2;
				end else begin
					rmiirxshim1_pre_cnt <= (rmiirxshim1_pre_cnt + 1'd1);
				end
			end
		end
	end
	if (((rmiirxshim1_mode == 2'd2) & rmiirxshim1_consume)) begin
		if (rmiirxshim1_pushing) begin
			rmiirxshim1_f0_valid <= 1'd1;
			rmiirxshim1_f0_data <= rmiirxshim1_f1_data;
			rmiirxshim1_f0_last <= rmiirxshim1_f1_last;
			rmiirxshim1_f1_valid <= 1'd1;
			rmiirxshim1_f1_data <= rmiirxshim1_new_byte;
			rmiirxshim1_f1_last <= rmiirxshim1_crs_last;
		end else begin
			rmiirxshim1_f0_valid <= rmiirxshim1_f1_valid;
			rmiirxshim1_f0_data <= rmiirxshim1_f1_data;
			rmiirxshim1_f0_last <= rmiirxshim1_f1_last;
			rmiirxshim1_f1_valid <= 1'd0;
		end
	end else begin
		if ((rmiirxshim1_pushing & (~(((rmiirxshim1_mode == 2'd2) & rmiirxshim1_f0_valid) & rmiirxshim1_f1_valid)))) begin
			if ((~rmiirxshim1_f0_valid)) begin
				rmiirxshim1_f0_valid <= 1'd1;
				rmiirxshim1_f0_data <= rmiirxshim1_new_byte;
				rmiirxshim1_f0_last <= rmiirxshim1_crs_last;
			end else begin
				if ((~rmiirxshim1_f1_valid)) begin
					rmiirxshim1_f1_valid <= 1'd1;
					rmiirxshim1_f1_data <= rmiirxshim1_new_byte;
					rmiirxshim1_f1_last <= rmiirxshim1_crs_last;
				end
			end
		end
	end
	if (((rmiirxshim1_mode != 1'd0) & rmiirxshim1_e1)) begin
		rmiirxshim1_shift <= rmiirxshim1_new_byte;
		if ((rmiirxshim1_bits == 2'd3)) begin
			rmiirxshim1_bits <= 1'd0;
		end else begin
			rmiirxshim1_bits <= (rmiirxshim1_bits + 1'd1);
		end
	end else begin
		if ((~rmiirxshim1_e1)) begin
			rmiirxshim1_bits <= 1'd0;
		end
	end
	if (negrx_rst) begin
		rmiirxshim1_crs_run <= 1'd0;
		rmiirxshim1_crs_dv_d <= 1'd0;
		rmiirxshim1_d0 <= 2'd0;
		rmiirxshim1_d1 <= 2'd0;
		rmiirxshim1_e0 <= 1'd0;
		rmiirxshim1_e1 <= 1'd0;
		rmiirxshim1_mode <= 2'd0;
		rmiirxshim1_f0_valid <= 1'd0;
		rmiirxshim1_f0_data <= 8'd0;
		rmiirxshim1_f0_last <= 1'd0;
		rmiirxshim1_f1_valid <= 1'd0;
		rmiirxshim1_f1_data <= 8'd0;
		rmiirxshim1_f1_last <= 1'd0;
		rmiirxshim1_sreg <= 8'd85;
		rmiirxshim1_pre_cnt <= 3'd0;
		rmiirxshim1_shift <= 8'd0;
		rmiirxshim1_bits <= 2'd0;
	end
end

always @(posedge sys_clk) begin
	pend_pos_d <= cap_pos_pend;
	pend_pos <= pend_pos_d;
	pend_neg_d <= cap_neg_pend;
	pend_neg <= pend_neg_d;
	len_pos <= cap_pos_frame_len;
	len_neg <= cap_neg_frame_len;
	if (hb_fire) begin
		hb <= 1'd1;
	end else begin
		hb <= (hb + 1'd1);
	end
	pend_tx_d <= cap_tx_pend;
	pend_tx <= pend_tx_d;
	len_tx <= cap_tx_frame_len;
	pend_tap_d <= cap_tap_pend;
	pend_tap <= pend_tap_d;
	len_tap <= cap_tap_frame_len;
	if ((state == 1'd0)) begin
		if (pend_pos) begin
			state <= 1'd1;
			marker <= 1'd1;
			length <= len_pos;
			idx <= 1'd0;
			clear_pos <= 1'd1;
		end else begin
			if (pend_neg) begin
				state <= 1'd1;
				marker <= 2'd2;
				length <= len_neg;
				idx <= 1'd0;
				clear_neg <= 1'd1;
			end else begin
				if (pend_tap) begin
					state <= 1'd1;
					marker <= 3'd4;
					length <= len_tap;
					idx <= 1'd0;
					clear_tap <= 1'd1;
				end else begin
					if (pend_tx) begin
						state <= 1'd1;
						marker <= 2'd3;
						length <= len_tx;
						idx <= 1'd0;
						clear_tx <= 1'd1;
					end
				end
			end
		end
	end else begin
		if ((state == 1'd1)) begin
			clear_pos <= 1'd0;
			clear_neg <= 1'd0;
			clear_tx <= 1'd0;
			clear_tap <= 1'd0;
			if ((~uarttx_busy)) begin
				state <= 2'd2;
			end
		end else begin
			if ((state == 2'd2)) begin
				if ((~uarttx_busy)) begin
					state <= 2'd3;
				end
			end else begin
				if ((state == 2'd3)) begin
					if ((~uarttx_busy)) begin
						if ((length == 1'd0)) begin
							state <= 1'd0;
						end else begin
							state <= 3'd4;
						end
					end
				end else begin
					if ((~uarttx_busy)) begin
						if ((idx == (length - 1'd1))) begin
							state <= 1'd0;
							idx <= 1'd0;
						end else begin
							idx <= (idx + 1'd1);
						end
					end
				end
			end
		end
	end
	cap_pos_clear <= cap_pos_clear_in;
	cap_pos_clear_d <= cap_pos_clear;
	if (cap_pos_clear_d) begin
		cap_pos_pend <= 1'd0;
	end else begin
		if ((rmiirxshim0_source_valid & rmiirxshim0_source_last)) begin
			cap_pos_pend <= 1'd1;
		end
	end
	if ((rmiirxshim0_source_valid & rmiirxshim0_source_last)) begin
		cap_pos_frame_len <= (cap_pos_wr_ptr + 1'd1);
		cap_pos_wr_ptr <= 1'd0;
	end else begin
		if ((rmiirxshim0_source_valid & (~cap_pos_pend))) begin
			if ((cap_pos_wr_ptr != 7'd127)) begin
				cap_pos_wr_ptr <= (cap_pos_wr_ptr + 1'd1);
			end
		end
	end
	cap_neg_clear <= cap_neg_clear_in;
	cap_neg_clear_d <= cap_neg_clear;
	if (cap_neg_clear_d) begin
		cap_neg_pend <= 1'd0;
	end else begin
		if ((rmiirxshim1_source_valid & rmiirxshim1_source_last)) begin
			cap_neg_pend <= 1'd1;
		end
	end
	if ((rmiirxshim1_source_valid & rmiirxshim1_source_last)) begin
		cap_neg_frame_len <= (cap_neg_wr_ptr + 1'd1);
		cap_neg_wr_ptr <= 1'd0;
	end else begin
		if ((rmiirxshim1_source_valid & (~cap_neg_pend))) begin
			if ((cap_neg_wr_ptr != 7'd127)) begin
				cap_neg_wr_ptr <= (cap_neg_wr_ptr + 1'd1);
			end
		end
	end
	cap_tx_clear <= cap_tx_clear_in;
	cap_tx_clear_d <= cap_tx_clear;
	if (cap_tx_clear_d) begin
		cap_tx_pend <= 1'd0;
	end else begin
		if ((txdibitpacker_source_valid & txdibitpacker_source_last)) begin
			cap_tx_pend <= 1'd1;
		end
	end
	if ((txdibitpacker_source_valid & txdibitpacker_source_last)) begin
		cap_tx_frame_len <= (cap_tx_wr_ptr + 1'd1);
		cap_tx_wr_ptr <= 1'd0;
	end else begin
		if ((txdibitpacker_source_valid & (~cap_tx_pend))) begin
			if ((cap_tx_wr_ptr != 7'd127)) begin
				cap_tx_wr_ptr <= (cap_tx_wr_ptr + 1'd1);
			end
		end
	end
	cap_tap_clear <= cap_tap_clear_in;
	cap_tap_clear_d <= cap_tap_clear;
	if (cap_tap_clear_d) begin
		cap_tap_pend <= 1'd0;
	end else begin
		if ((txtap_source_valid & txtap_source_last)) begin
			cap_tap_pend <= 1'd1;
		end
	end
	if ((txtap_source_valid & txtap_source_last)) begin
		cap_tap_frame_len <= (cap_tap_wr_ptr + 1'd1);
		cap_tap_wr_ptr <= 1'd0;
	end else begin
		if ((txtap_source_valid & (~cap_tap_pend))) begin
			if ((cap_tap_wr_ptr != 7'd127)) begin
				cap_tap_wr_ptr <= (cap_tap_wr_ptr + 1'd1);
			end
		end
	end
	if ((uarttx_state == 1'd0)) begin
		if (uarttx_strobe) begin
			uarttx_state <= 1'd1;
			uarttx_cnt <= 1'd0;
			uarttx_shreg <= uarttx_data;
			uarttx_bitidx <= 1'd0;
		end else begin
			if (uarttx_pend_bit) begin
				uarttx_state <= 1'd1;
				uarttx_cnt <= 1'd0;
				uarttx_shreg <= uarttx_pend_data;
				uarttx_pend_bit <= 1'd0;
				uarttx_bitidx <= 1'd0;
			end
		end
	end else begin
		if ((uarttx_state == 1'd1)) begin
			uarttx_cnt <= (uarttx_cnt + 1'd1);
			if ((uarttx_cnt == 9'd433)) begin
				uarttx_state <= 2'd2;
				uarttx_cnt <= 1'd0;
			end
		end else begin
			if ((uarttx_state == 2'd2)) begin
				uarttx_cnt <= (uarttx_cnt + 1'd1);
				if ((uarttx_cnt == 9'd433)) begin
					uarttx_shreg <= {uarttx, uarttx_shreg[7:1]};
					if ((uarttx_bitidx == 3'd7)) begin
						uarttx_state <= 2'd3;
					end else begin
						uarttx_bitidx <= (uarttx_bitidx + 1'd1);
					end
					uarttx_cnt <= 1'd0;
				end
			end else begin
				uarttx_cnt <= (uarttx_cnt + 1'd1);
				if ((uarttx_cnt == 9'd433)) begin
					uarttx_state <= 1'd0;
					uarttx_cnt <= 1'd0;
				end
			end
		end
	end
	if ((uarttx_strobe & (uarttx_state != 1'd0))) begin
		uarttx_pend_bit <= 1'd1;
		uarttx_pend_data <= uarttx_data;
	end
	if (sys_rst) begin
		cap_pos_wr_ptr <= 7'd0;
		cap_pos_pend <= 1'd0;
		cap_pos_clear_d <= 1'd0;
		cap_pos_clear <= 1'd0;
		cap_pos_frame_len <= 7'd0;
		cap_neg_wr_ptr <= 7'd0;
		cap_neg_pend <= 1'd0;
		cap_neg_clear_d <= 1'd0;
		cap_neg_clear <= 1'd0;
		cap_neg_frame_len <= 7'd0;
		cap_tx_wr_ptr <= 7'd0;
		cap_tx_pend <= 1'd0;
		cap_tx_clear_d <= 1'd0;
		cap_tx_clear <= 1'd0;
		cap_tx_frame_len <= 7'd0;
		cap_tap_wr_ptr <= 7'd0;
		cap_tap_pend <= 1'd0;
		cap_tap_clear_d <= 1'd0;
		cap_tap_clear <= 1'd0;
		cap_tap_frame_len <= 7'd0;
		uarttx_state <= 2'd0;
		uarttx_cnt <= 10'd0;
		uarttx_shreg <= 8'd0;
		uarttx_bitidx <= 3'd0;
		uarttx_pend_bit <= 1'd0;
		uarttx_pend_data <= 8'd0;
		pend_pos_d <= 1'd0;
		pend_neg_d <= 1'd0;
		pend_pos <= 1'd0;
		pend_neg <= 1'd0;
		len_pos <= 7'd0;
		len_neg <= 7'd0;
		hb <= 26'd0;
		state <= 3'd0;
		marker <= 8'd0;
		length <= 7'd0;
		idx <= 7'd0;
		clear_pos <= 1'd0;
		clear_neg <= 1'd0;
		clear_tx <= 1'd0;
		clear_tap <= 1'd0;
		pend_tx_d <= 1'd0;
		pend_tx <= 1'd0;
		len_tx <= 7'd0;
		pend_tap_d <= 1'd0;
		pend_tap <= 1'd0;
		len_tap <= 7'd0;
	end
end

reg [7:0] cap_pos[0:127];
reg [7:0] memdat;
always @(posedge eth_rx_clk) begin : mem_write_block
	if (cap_pos_wr_we)
		cap_pos[cap_pos_wr_adr] <= cap_pos_wr_dat_w;
	memdat <= cap_pos[cap_pos_wr_adr];
end

always @(posedge sys_clk) begin : mem_write_block_1
end

assign cap_pos_wr_dat_r = memdat;
assign cap_pos_rd_port_dat_r = cap_pos[cap_pos_rd_port_adr];

reg [7:0] cap_neg[0:127];
reg [7:0] memdat_1;
always @(posedge negrx_clk) begin : mem_write_block_2
	if (cap_neg_wr_we)
		cap_neg[cap_neg_wr_adr] <= cap_neg_wr_dat_w;
	memdat_1 <= cap_neg[cap_neg_wr_adr];
end

always @(posedge sys_clk) begin : mem_write_block_3
end

assign cap_neg_wr_dat_r = memdat_1;
assign cap_neg_rd_port_dat_r = cap_neg[cap_neg_rd_port_adr];

reg [7:0] cap_tx[0:127];
reg [7:0] memdat_2;
always @(posedge eth_tx_clk) begin : mem_write_block_4
	if (cap_tx_wr_we)
		cap_tx[cap_tx_wr_adr] <= cap_tx_wr_dat_w;
	memdat_2 <= cap_tx[cap_tx_wr_adr];
end

always @(posedge sys_clk) begin : mem_write_block_5
end

assign cap_tx_wr_dat_r = memdat_2;
assign cap_tx_rd_port_dat_r = cap_tx[cap_tx_rd_port_adr];

reg [7:0] cap_tap[0:127];
reg [7:0] memdat_3;
always @(posedge eth_tx_clk) begin : mem_write_block_6
	if (cap_tap_wr_we)
		cap_tap[cap_tap_wr_adr] <= cap_tap_wr_dat_w;
	memdat_3 <= cap_tap[cap_tap_wr_adr];
end

always @(posedge sys_clk) begin : mem_write_block_7
end

assign cap_tap_wr_dat_r = memdat_3;
assign cap_tap_rd_port_dat_r = cap_tap[cap_tap_rd_port_adr];

endmodule
